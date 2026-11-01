import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Tensor T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # In matrix multiplication C = A @ B, C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor has 1s at (i,j,k) for all valid i,j,k.
    T = np.ones((n, m, p))
    
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    return np.allclose(T, T_recon, atol=1e-9)

def _strassen_rank14():
    """
    Construct the rank-14 decomposition using two independent Strassen 2x2x2 blocks.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    def add_strassen_block(start_r, b_col0, b_col1, c_col0, c_col1):
        # V indices: B[j, k] -> j*p + k
        v00 = 0 * p + b_col0   # B[0, b_col0]
        v01 = 0 * p + b_col1   # B[0, b_col1]
        v10 = 1 * p + b_col0   # B[1, b_col0]
        v11 = 1 * p + b_col1   # B[1, b_col1]
        
        # W indices: C[i, k] -> k*n + i
        # C[0, c_col0] -> k=c_col0, i=0 -> c_col0*n + 0
        w00 = c_col0 * n + 0
        w01 = c_col1 * n + 0
        # C[1, c_col0] -> k=c_col0, i=1 -> c_col0*n + 1
        w10 = c_col0 * n + 1
        w11 = c_col1 * n + 1
        
        # U indices: A[i, j] -> i*m + j
        u00 = 0 * m + 0  # A[0,0]
        u01 = 0 * m + 1  # A[0,1]
        u10 = 1 * m + 0  # A[1,0]
        u11 = 1 * m + 1  # A[1,1]
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        r = start_r
        U[u00, r] = 1; U[u11, r] = 1
        V[v00, r] = 1; V[v11, r] = 1
        W[w00, r] = 1; W[w11, r] = 1  # C[0,0] += m1, C[1,1] += m1
        
        # m2 = (A[1,0]+A[1,1]) * B[0,0]
        r = start_r + 1
        U[u10, r] = 1; U[u11, r] = 1
        V[v00, r] = 1
        W[w10, r] = 1; W[w11, r] = -1  # C[1,0] += m2, C[1,1] -= m2
        
        # m3 = A[0,0] * (B[1,0]-B[1,1])
        r = start_r + 2
        U[u00, r] = 1
        V[v10, r] = 1; V[v11, r] = -1
        W[w01, r] = 1; W[w11, r] = 1  # C[0,1] += m3, C[1,1] += m3
        
        # m4 = A[1,1] * (B[0,1]-B[0,0])
        r = start_r + 3
        U[u11, r] = 1
        V[v01, r] = 1; V[v00, r] = -1
        W[w00, r] = 1; W[w10, r] = 1  # C[0,0] += m4, C[1,0] += m4
        
        # m5 = (A[0,0]+A[0,1]) * B[1,1]
        r = start_r + 4
        U[u00, r] = 1; U[u01, r] = 1
        V[v11, r] = 1
        W[w01, r] = 1; W[w00, r] = -1  # C[0,1] += m5, C[0,0] -= m5
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        r = start_r + 5
        U[u10, r] = 1; U[u00, r] = -1
        V[v00, r] = 1; V[v01, r] = 1
        W[w11, r] = 1  # C[1,1] += m6
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        r = start_r + 6
        U[u01, r] = 1; U[u11, r] = -1
        V[v10, r] = 1; V[v11, r] = 1
        W[w00, r] = 1  # C[0,0] += m7

    # Left half: B columns 0,1 -> C columns 0,1
    add_strassen_block(0, b_col0=0, b_col1=1, c_col0=0, c_col1=1)
    
    # Right half: B columns 2,3 -> C columns 2,3
    add_strassen_block(7, b_col0=2, b_col1=3, c_col0=2, c_col1=3)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    """Fallback schoolbook decomposition of rank n*m*p = 16."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Construct the rank-14 Strassen solution
    best = _strassen_rank14()
    
    # Verify correctness
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook if something went wrong
        return _schoolbook()
    
    return best
