import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0
    
    # Strategy: Partition 6x6 into 2x2 blocks of 3x3 submatrices
    # Use schoolbook (rank 27) for each 3x3 block multiplication
    # 4 block positions x rank 27 = total rank 108
    
    block_size = 3  # 3x3 blocks
    blocks_per_dim = 2  # 2x2 grid of blocks, 2*3=6
    
    rank = (blocks_per_dim ** 2) * (block_size ** 3)  # 4 * 27 = 108
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for I in range(blocks_per_dim):  # block row of A and C
        for K in range(blocks_per_dim):  # block col of A, block row of B
            for J in range(blocks_per_dim):  # block col of B and C
                # Schoolbook for 3x3 block multiplication
                for i in range(block_size):  # local row in block
                    for k in range(block_size):  # local col in A block / row in B block
                        for j in range(block_size):  # local col in B block
                            # Global indices
                            a_row = I * block_size + i
                            a_col = K * block_size + k
                            b_row = K * block_size + k
                            b_col = J * block_size + j
                            c_row = I * block_size + i
                            c_col = J * block_size + j
                            
                            U[a_row * m + a_col, idx] = 1
                            V[b_row * p + b_col, idx] = 1
                            # W index: k*n + i where k is col of C, i is row of C
                            W[c_col * n + c_row, idx] = 1
                            idx += 1
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0.0
                for t in range(rank):
                    s = 0.0
                    for k in range(m):
                        s += U_arr[i*m+k, t] * V_arr[k*p+j, t]
                    total += W_arr[j*n+i, t] * s
                C_computed[i, j] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)
            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
