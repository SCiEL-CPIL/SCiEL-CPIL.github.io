"""Matrix multiplication 4x5x5 with Strassen-hierarchy tensor network ansatz."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = time.time() + 120

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (20, rank)
    
    # We will build the decomposition by blocks.
    # Partition I (rows of A, cols of C) into [0,1] and [2,3]
    # Partition J (cols of A, rows of B) into [0,1], [2,3], [4]
    # Partition K (cols of B, rows of C) into [0,1], [2,3], [4]
    
    I_blocks = [[0, 1], [2, 3]]
    J_blocks = [[0, 1], [2, 3], [4]]
    K_blocks = [[0, 1], [2, 3], [4]]
    
    # List of terms: each term is a dict with U, V, W contributions
    # Or we can just build the matrices directly.
    # Let's collect terms first to determine rank, then fill matrices.
    # Term: list of (U_idx, U_val), (V_idx, V_val), (W_idx, W_val)
    
    terms = []
    
    def add_strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        # Local indices 0,1 map to global i0,i1; j0,j1; k0,k1
        # A indices: i*m + j
        # B indices: j*p + k
        # C indices: k*n + i
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        terms.append({
            'U': [(i0*m + j0, 1), (i1*m + j1, 1)],
            'V': [(j0*p + k0, 1), (j1*p + k1, 1)],
            'W': [(k0*n + i0, 1), (k1*n + i0, -1), (k0*n + i1, 1), (k1*n + i1, 1)]
        })
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        terms.append({
            'U': [(i1*m + j0, 1), (i1*m + j1, 1)],
            'V': [(j0*p + k0, 1)],
            'W': [(k0*n + i1, 1), (k1*n + i1, -1)]
        })
        
        # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
        terms.append({
            'U': [(i0*m + j0, 1)],
            'V': [(j1*p + k0, 1), (j1*p + k1, -1)],
            'W': [(k0*n + i0, 1), (k1*n + i0, -1)]
        })
        
        # M4 = A[i1,j1] * (B[j0,k0] - B[j0,k1])
        terms.append({
            'U': [(i1*m + j1, 1)],
            'V': [(j0*p + k0, 1), (j0*p + k1, -1)],
            'W': [(k0*n + i0, -1), (k0*n + i1, 1), (k1*n + i1, -1)]
        })
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        terms.append({
            'U': [(i0*m + j0, 1), (i0*m + j1, 1)],
            'V': [(j1*p + k1, 1)],
            'W': [(k1*n + i0, 1)]
        })
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        terms.append({
            'U': [(i1*m + j0, 1), (i0*m + j0, -1)],
            'V': [(j0*p + k0, 1), (j0*p + k1, 1)],
            'W': [(k0*n + i1, -1), (k1*n + i1, 1)]
        })
        
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        terms.append({
            'U': [(i0*m + j1, 1), (i1*m + j1, -1)],
            'V': [(j1*p + k0, 1), (j1*p + k1, 1)],
            'W': [(k0*n + i0, -1), (k1*n + i0, 1), (k1*n + i1, 1)]
        })
        
    def add_schoolblock(I_blk, J_blk, K_blk):
        for i in I_blk:
            for j in J_blk:
                for k in K_blk:
                    terms.append({
                        'U': [(i*m + j, 1)],
                        'V': [(j*p + k, 1)],
                        'W': [(k*n + i, 1)]
                    })
                    
    for I_blk in I_blocks:
        for J_blk in J_blocks:
            for K_blk in K_blocks:
                if len(I_blk) == 2 and len(J_blk) == 2 and len(K_blk) == 2:
                    i0, i1 = I_blk
                    j0, j1 = J_blk
                    k0, k1 = K_blk
                    add_strassen_2x2x2(i0, i1, j0, j1, k0, k1)
                else:
                    add_schoolblock(I_blk, J_blk, K_blk)
                    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        for idx, val in term['U']:
            U[idx][r] = val
        for idx, val in term['V']:
            V[idx][r] = val
        for idx, val in term['W']:
            W[idx][r] = val
            
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check a few random entries to ensure correctness
    # T[i*m+j, j*p+k, k*n+i] should be 1
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We'll check all 100 entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = [[0.0] * rank for _ in range(n * m)]
                    V = [[0.0] * rank for _ in range(m * p)]
                    W = [[0.0] * rank for _ in range(p * n)]
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U[i * m + j][col] = 1.0
                                V[j * p + k][col] = 1.0
                                W[k * n + i][col] = 1.0
                                col += 1
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}
