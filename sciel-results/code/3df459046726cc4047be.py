"""
Solution for 4x4 matrix multiplication with minimal rank 49.

Uses a 2-level Strassen decomposition (Kronecker product of two 2x2 Strassen decompositions).
The rank is 7 * 7 = 49.
"""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    n, m, p = 4, 4, 4
    rank = 49

    # --- 2x2 Strassen Coefficients ---
    # U_elem[t] is a vector of 4 coefficients for the t-th product, corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    U_elem = [
        [1, 0, 0, 1],    # t=0: A00 + A11
        [0, 0, 1, 1],    # t=1: A10 + A11
        [1, 0, 0, 0],    # t=2: A00
        [0, 0, 0, 1],    # t=3: A11
        [1, 1, 0, 0],    # t=4: A00 + A01
        [-1, 0, 1, 0],   # t=5: A10 - A00
        [0, 1, 0, -1]    # t=6: A01 - A11
    ]
    # V_elem[t] is a vector of 4 coefficients for the t-th product, corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    V_elem = [
        [1, 0, 0, 1],    # t=0: B00 + B11
        [1, 0, 0, 0],    # t=1: B00
        [0, 1, 0, -1],   # t=2: B01 - B11
        [-1, 0, 1, 0],   # t=3: B10 - B00
        [0, 0, 0, 1],    # t=4: B11
        [1, 1, 0, 0],    # t=5: B00 + B01
        [0, 0, 1, 1]     # t=6: B10 + B11
    ]
    # W_elem[t] is a 2x2 matrix of coefficients for the t-th product.
    # W_elem[t][i][k] is the coefficient for C[i,k] (row i, col k)
    W_elem = [
        [[1, 0], [0, 1]],   # t=0: C00+=M, C11+=M
        [[0, 0], [1, -1]],  # t=1: C10+=M, C11-=M
        [[0, 1], [0, 1]],   # t=2: C01+=M, C11+=M
        [[1, 0], [1, 0]],   # t=3: C00+=M, C10+=M
        [[-1, 1], [0, 0]],  # t=4: C00-=M, C01+=M
        [[0, 0], [0, 1]],   # t=5: C11+=M
        [[1, 0], [0, 0]]    # t=6: C00+=M
    ]

    # --- Block Level Strassen Coefficients ---
    # A_blocks[s]: list of (block_row, block_col, coeff)
    # Block indices are 0,1 for 2x2 blocks
    A_blocks = [
        [(0, 0, 1), (1, 1, 1)],      # s=0: A00 + A11
        [(1, 0, 1), (1, 1, 1)],      # s=1: A10 + A11
        [(0, 0, 1)],                 # s=2: A00
        [(1, 1, 1)],                 # s=3: A11
        [(0, 0, 1), (0, 1, 1)],      # s=4: A00 + A01
        [(1, 0, 1), (0, 0, -1)],     # s=5: A10 - A00
        [(0, 1, 1), (1, 1, -1)]      # s=6: A01 - A11
    ]
    # B_blocks[s]: list of (block_col, block_col2, coeff) i.e., (J, K, coeff)
    B_blocks = [
        [(0, 0, 1), (1, 1, 1)],      # s=0: B00 + B11
        [(0, 0, 1)],                 # s=1: B00
        [(0, 1, 1), (1, 1, -1)],     # s=2: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # s=3: B10 - B00
        [(1, 1, 1)],                 # s=4: B11
        [(0, 0, 1), (0, 1, 1)],      # s=5: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # s=6: B10 + B11
    ]
    # C_blocks[s]: list of (block_row, block_col, coeff) i.e., (I, K, coeff)
    C_blocks = [
        [(0, 0, 1), (1, 1, 1)],      # s=0: C00+=M, C11+=M
        [(1, 0, 1), (1, 1, -1)],     # s=1: C10+=M, C11-=M
        [(0, 1, 1), (1, 1, 1)],      # s=2: C01+=M, C11+=M
        [(0, 0, 1), (1, 0, 1)],      # s=3: C00+=M, C10+=M
        [(0, 0, -1), (0, 1, 1)],     # s=4: C00-=M, C01+=M
        [(1, 1, 1)],                 # s=5: C11+=M
        [(0, 0, 1)]                  # s=6: C00+=M
    ]

    # Initialize U, V, W matrices
    # U: (16, 49), V: (16, 49), W: (16, 49)
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Build U contributions
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        # Element in A at global row bi*2+ei, col bj*2+ej
                        row_a = (bi * 2 + ei) * 4 + (bj * 2 + ej)
                        # Coefficient is product of block coeff and element coeff
                        # Element index in U_elem is ei*2+ej
                        U[row_a, col] += c_a * U_elem[t][ei * 2 + ej]
            
            # Build V contributions
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        # Element in B at global row bj*2+ej, col bk*2+ek
                        row_b = (bj * 2 + ej) * 4 + (bk * 2 + ek)
                        # Element index in V_elem is ej*2+ek
                        V[row_b, col] += c_b * V_elem[t][ej * 2 + ek]
            
            # Build W contributions
            # C_blocks[s] gives (bi, bk, coeff) meaning C block (bi, bk) gets coeff * W_elem
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        # Element in C at global row bi*2+ei, col bk*2+ek
                        # Task requires W[k*n+i, r]. Here n=4.
                        # Global row index i = bi*2+ei
                        # Global col index k = bk*2+ek
                        # W row index = k*4 + i
                        row_c = (bk * 2 + ek) * 4 + (bi * 2 + ei)
                        # W_elem[t] is 2x2 matrix, index [ei][ek] corresponds to local C[ei, ek]
                        # But W_elem is defined as W_elem[t][i][k] for C[i,k]
                        # So we use W_elem[t][ei][ek]
                        W[row_c, col] += c_c * W_elem[t][ei][ek]

    # Verify correctness using vectorized tensor contraction
    def verify(U, V, W, n, m, p, rank):
        if time.time() - start_time > deadline - 10:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Construct the target tensor T_{u v w}
        # u = i * m + j
        # v = j * p + k
        # w = k * n + i
        # T_{u v w} = 1 if j_u == j_v and i_u == i_w and k_v == k_w else 0
        
        # We can build T by iterating over valid (i, j, k) tuples
        # There are n * m * p = 64 such tuples.
        # For each (i, j, k), u = i*m+j, v = j*p+k, w = k*n+i
        # T[u, v, w] = 1
        
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        
        # Compute decomposed tensor D_{u v w} = sum_r U[u, r] * V[v, r] * W[w, r]
        # Using einsum for speed
        D = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
        
        if not np.allclose(T, D, atol=1e-9):
            return False
        
        return True

    if verify(U, V, W, n, m, p, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 64 if verification fails (should not happen)
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=int)
        V_fb = np.zeros((16, rank_fb), dtype=int)
        W_fb = np.zeros((16, rank_fb), dtype=int)
        col_fb = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col_fb] = 1
                    V_fb[j*4+k, col_fb] = 1
                    W_fb[k*4+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
