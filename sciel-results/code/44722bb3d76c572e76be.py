"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Uses Strassen optimization on 2x2x2 sub-blocks to reduce rank below schoolbook 63.
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    
    # Indexing:
    # U index: i*m + j for A[i,j]
    # V index: j*p + k for B[j,k]
    # W index: k*n + i for C[i,k] (trace convention: W[k*n+i, r])
    
    terms = []  # Each term: list of (u_idx, coeff), list of (v_idx, coeff), list of (w_idx, coeff)
    
    def add_strassen_term(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for a 2x2x2 sub-block."""
        # Term 1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1]) -> C[i0,k0] + C[i1,k1]
        u1 = [(i0*m+j0, 1), (i1*m+j1, 1)]
        v1 = [(j0*p+k0, 1), (j1*p+k1, 1)]
        w1 = [(k0*n+i0, 1), (k1*n+i1, 1)]
        terms.append((u1, v1, w1))
        
        # Term 2: (A[i1,j0]+A[i1,j1]) * B[j0,k0] -> C[i1,k0]
        u2 = [(i1*m+j0, 1), (i1*m+j1, 1)]
        v2 = [(j0*p+k0, 1)]
        w2 = [(k0*n+i1, 1)]
        terms.append((u2, v2, w2))
        
        # Term 3: A[i0,j0] * (B[j1,k0]+B[j1,k1]) -> C[i0,k0]
        u3 = [(i0*m+j0, 1)]
        v3 = [(j1*p+k0, 1), (j1*p+k1, 1)]
        w3 = [(k0*n+i0, 1)]
        terms.append((u3, v3, w3))
        
        # Term 4: (A[i0,j0]-A[i1,j0]) * (B[j1,k1]-B[j0,k0]) -> C[i0,k0] + C[i1,k1]
        u4 = [(i0*m+j0, 1), (i1*m+j0, -1)]
        v4 = [(j1*p+k1, 1), (j0*p+k0, -1)]
        w4 = [(k0*n+i0, 1), (k1*n+i1, 1)]
        terms.append((u4, v4, w4))
        
        # Term 5: (A[i0,j0]+A[i0,j1]) * B[j1,k1] -> C[i0,k1]
        u5 = [(i0*m+j0, 1), (i0*m+j1, 1)]
        v5 = [(j1*p+k1, 1)]
        w5 = [(k1*n+i0, 1)]
        terms.append((u5, v5, w5))
        
        # Term 6: A[i1,j1] * (B[j0,k0]-B[j1,k0]) -> C[i1,k0]
        u6 = [(i1*m+j1, 1)]
        v6 = [(j0*p+k0, 1), (j1*p+k0, -1)]
        w6 = [(k0*n+i1, 1)]
        terms.append((u6, v6, w6))
        
        # Term 7: (A[i0,j1]-A[i1,j1]) * (B[j0,k0]+B[j1,k0]) -> -C[i0,k0] + C[i0,k1]
        u7 = [(i0*m+j1, 1), (i1*m+j1, -1)]
        v7 = [(j0*p+k0, 1), (j1*p+k0, 1)]
        w7 = [(k0*n+i0, -1), (k1*n+i0, 1)]
        terms.append((u7, v7, w7))
    
    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook term."""
        terms.append([[(i*m+j, 1)], [(j*p+k, 1)], [(k*n+i, 1)]])
    
    # Apply Strassen to 2x2x2 sub-blocks where possible
    # i in {0,1}, j in {0,1}, k in {0,1}: Strassen
    add_strassen_term(0, 1, 0, 1, 0, 1)
    # i in {0,1}, j in {0,1}, k in {2,3}: Strassen
    add_strassen_term(0, 1, 0, 1, 2, 3)
    # i in {0,1}, j in {0,1}, k in {4,5}: Strassen
    add_strassen_term(0, 1, 0, 1, 4, 5)
    
    # Remaining terms (schoolbook):
    # i in {0,1}, j in {0,1}, k=6
    for i in [0, 1]:
        for j in [0, 1]:
            add_schoolbook_term(i, j, 6)
    
    # i in {0,1}, j=2, k in 0..6
    for i in [0, 1]:
        for k in range(p):
            add_schoolbook_term(i, 2, k)
    
    # i=2, j in {0,1}, k in 0..6
    for j in [0, 1]:
        for k in range(p):
            add_schoolbook_term(2, j, k)
    
    # i=2, j=2, k in 0..6
    for k in range(p):
        add_schoolbook_term(2, 2, k)
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for idx, coeff in u_list:
            U[idx][r] = coeff
        for idx, coeff in v_list:
            V[idx][r] = coeff
        for idx, coeff in w_list:
            W[idx][r] = coeff
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    # Actually, the tensor is T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 for all valid (i,j,k)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
