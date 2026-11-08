import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: Use rank-4 decomposition
    # For each j in {0,1,2,3}, use one rank-1 term
    # U[i*m+j, r_j] = 1 for all i
    # V[j*p+k, r_j] = 1 for all k  
    # W[k*n+i, r_j] = 1 for all k,i
    # This gives sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    rank = 4
    U = [[0] * rank for _ in range(n * m)]  # 8 x 4
    V = [[0] * rank for _ in range(m * p)]  # 32 x 4
    W = [[0] * rank for _ in range(p * n)]  # 16 x 4
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    # Verify the rank-4 decomposition
    def full_verify():
        import numpy as np
        # Test with random matrices
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute C from decomposition: C[i,k] = sum_j sum_r A[i,j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    for r in range(rank):
                        total += A[i, j] * U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = total
        
        return np.allclose(C_computed, C_expected)
    
    if full_verify():
        return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-8 construction using 2 terms per j
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use 2 terms per j: one for i=0, one for i=1
    for j in range(m):
        for i in range(n):
            r = j * 2 + i
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    if full_verify():
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: Strassen-based rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)],
    ]
    
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                U[(li + i_off) * 4 + (lj + j_off)][cr] += val
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                W[(lk + k_off) * 2 + (li + i_off)][cr] += val
    
    for j_block in range(2):
        for k_block in range(4):
            base = (j_block * 4 + k_block) * 7
            add_strassen_block(base, 0, j_block * 2, k_block * 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
