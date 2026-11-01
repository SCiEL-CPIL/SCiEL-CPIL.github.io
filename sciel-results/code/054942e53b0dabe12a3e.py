import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2x2
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = num_blocks * 7  # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms: (a, b, c, d, e, f, g, h, C00, C01, C10, C11)
    strassen_terms = [
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),   # M1
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),   # M2
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),   # M3
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),   # M4
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),   # M5
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),   # M6
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),   # M7
    ]
    
    for block in range(num_blocks):
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c00_c, c01_c, c10_c, c11_c = term
            
            # A part: U[i*m+j, col]
            U[0][col] = a_c   # A[0,0]
            U[1][col] = b_c   # A[0,1]
            U[2][col] = c_c   # A[1,0]
            U[3][col] = d_c   # A[1,1]
            
            # B part: V[j*p+k, col]
            V[0 * p + (2 * block)][col] = e_c
            V[0 * p + (2 * block + 1)][col] = f_c
            V[1 * p + (2 * block)][col] = g_c
            V[1 * p + (2 * block + 1)][col] = h_c
            
            # C part: W[k*n+i, col]
            W[(2 * block) * n + 0][col] = c00_c
            W[(2 * block + 1) * n + 0][col] = c01_c
            W[(2 * block) * n + 1][col] = c10_c
            W[(2 * block + 1) * n + 1][col] = c11_c
    
    # Verify correctness with a random test
    if time.time() < deadline:
        np.random.seed(42)
        A = np.random.randint(1, 5, size=(n, m)).astype(float)
        B = np.random.randint(1, 5, size=(m, p)).astype(float)
        C_expected = A @ B
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Compute C from decomposition: C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # More precisely: C = (U^T * A_vec) * (V^T * B_vec) * W^T
        # Actually: C[i,k] = sum_{t} sum_{i',j'} U[i'*m+j', t] * A[i',j'] * V[j'*p+k, t] * W[k*n+i, t]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for t in range(rank):
                    # Sum over A entries
                    a_val = 0.0
                    for ii in range(n):
                        for jj in range(m):
                            a_val += U_np[ii * m + jj, t] * A[ii, jj]
                    # Sum over B entries
                    b_val = 0.0
                    for jj in range(m):
                        b_val += V_np[jj * p + k, t] * B[jj, k]
                    total += a_val * b_val * W_np[k * n + i, t]
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to trivial rank-56 decomposition
            rank_trivial = n * m * p
            U_t = [[0] * rank_trivial for _ in range(n * m)]
            V_t = [[0] * rank_trivial for _ in range(m * p)]
            W_t = [[0] * rank_trivial for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_t[i * m + j][col] = 1
                        V_t[j * p + k][col] = 1
                        W_t[k * n + i][col] = 1
                        col += 1
            return {"U": U_t, "V": V_t, "W": W_t}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
