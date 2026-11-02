"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

TIME_LIMIT = 140.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # Strassen-based decomposition for 3x4 x 4x4 -> 3x4
    # Total rank: 4 Strassen blocks (7 each) + 4 naive = 32
    
    rank = 32
    U = np.zeros((n*m, rank))  # (12, 32)
    V = np.zeros((m*p, rank))  # (16, 32)
    W = np.zeros((p*n, rank))  # (12, 32)
    
    col = 0
    
    def add_strassen(a_i1, a_i2, a_j1, a_j2, b_j1, b_j2, b_k1, b_k2, c_i1, c_i2, c_k1, c_k2, sign=1.0):
        """Add Strassen decomposition for 2x2x2 block.
        A block: rows a_i1,a_i2 and cols a_j1,a_j2
        B block: rows b_j1,b_j2 and cols b_k1,b_k2
        C block: rows c_i1,c_i2 and cols c_k1,c_k2
        """
        nonlocal col
        # U indices: i*m + j
        a00_idx = a_i1*m + a_j1
        a01_idx = a_i1*m + a_j2
        a10_idx = a_i2*m + a_j1
        a11_idx = a_i2*m + a_j2
        # V indices: j*p + k
        b00_idx = b_j1*p + b_k1
        b01_idx = b_j1*p + b_k2
        b10_idx = b_j2*p + b_k1
        b11_idx = b_j2*p + b_k2
        # W indices: k*n + i
        c00_idx = c_k1*n + c_i1
        c01_idx = c_k2*n + c_i1
        c10_idx = c_k1*n + c_i2
        c11_idx = c_k2*n + c_i2
        
        # p1 = (a00+a11)(b00+b11) -> c00 += p1, c11 += p1
        U[a00_idx, col] += sign
        U[a11_idx, col] += sign
        V[b00_idx, col] += sign
        V[b11_idx, col] += sign
        W[c00_idx, col] += sign
        W[c11_idx, col] += sign
        col += 1
        
        # p2 = (a10+a11)b00 -> c10 += p2, c11 -= p2
        U[a10_idx, col] += sign
        U[a11_idx, col] += sign
        V[b00_idx, col] += sign
        W[c10_idx, col] += sign
        W[c11_idx, col] -= sign
        col += 1
        
        # p3 = a00(b01-b11) -> c01 += p3, c11 += p3
        U[a00_idx, col] += sign
        V[b01_idx, col] += sign
        V[b11_idx, col] -= sign
        W[c01_idx, col] += sign
        W[c11_idx, col] += sign
        col += 1
        
        # p4 = a11(b10-b00) -> c00 += p4, c10 += p4
        U[a11_idx, col] += sign
        V[b10_idx, col] += sign
        V[b00_idx, col] -= sign
        W[c00_idx, col] += sign
        W[c10_idx, col] += sign
        col += 1
        
        # p5 = (a00+a01)b11 -> c00 -= p5, c01 += p5
        U[a00_idx, col] += sign
        U[a01_idx, col] += sign
        V[b11_idx, col] += sign
        W[c00_idx, col] -= sign
        W[c01_idx, col] += sign
        col += 1
        
        # p6 = (a10-a00)(b00+b01) -> c11 += p6
        U[a10_idx, col] += sign
        U[a00_idx, col] -= sign
        V[b00_idx, col] += sign
        V[b01_idx, col] += sign
        W[c11_idx, col] += sign
        col += 1
        
        # p7 = (a01-a11)(b10+b11) -> c00 += p7
        U[a01_idx, col] += sign
        U[a11_idx, col] -= sign
        V[b10_idx, col] += sign
        V[b11_idx, col] += sign
        W[c00_idx, col] += sign
        col += 1
    
    # Block 1: C[0:2, 0:2] += A[0:2, 0:2] @ B[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, sign=1.0)
    
    # Block 2: C[0:2, 0:2] += A[0:2, 2:4] @ B[2:4, 0:2]
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, sign=1.0)
    
    # Block 3: C[0:2, 2:4] += A[0:2, 0:2] @ B[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, sign=1.0)
    
    # Block 4: C[0:2, 2:4] += A[0:2, 2:4] @ B[2:4, 2:4]
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, sign=1.0)
    
    # Bottom row: C[2,k] = sum_j A[2,j] * B[j,k] for k=0,1,2,3
    for k in range(4):
        for j in range(4):
            U[2*m + j, col] = 1.0
            V[j*p + k, col] = 1.0
            W[k*n + 2, col] = 1.0
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, rnk):
        reconstructed = np.zeros((n, m, p))
        for r in range(rnk):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        reconstructed[i, j, k] += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
        target = np.ones((n, m, p))
        return np.allclose(reconstructed, target, atol=1e-10)
    
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((n*m, rank_sb))
        V = np.zeros((m*p, rank_sb))
        W = np.zeros((p*n, rank_sb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        rank = rank_sb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
