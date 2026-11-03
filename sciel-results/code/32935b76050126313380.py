import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    t0 = time.time()
    deadline = t0 + 130.0
    
    # Rank 28 Strassen solution
    def build_strassen():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen steps for 2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We have 4 blocks:
        # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # 2. A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
        # 3. A[2:4, 0:2] * B[2:4, 0:2] -> C[2:4, 0:2]
        # 4. A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
        # Wait, A is 2x4. So A[2:4] is empty?
        # A is 2x4. Rows 0,1. Cols 0,1,2,3.
        # B is 4x4. Rows 0,1,2,3. Cols 0,1,2,3.
        # C is 2x4. Rows 0,1. Cols 0,1,2,3.
        #
        # C[i,k] = sum_j A[i,j] B[j,k]
        #
        # The current code's add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset)
        # a_col_start: start col of A block (2x2)
        # b_row_start: start row of B block (2x2)
        # b_col_start: start col of B block (2x2)
        # c_col_start: start col of C block (2x2)
        #
        # Block 1: A cols 0-1, B rows 0-1, B cols 0-1 -> C cols 0-1
        # Block 2: A cols 0-1, B rows 0-1, B cols 2-3 -> C cols 2-3
        # Block 3: A cols 2-3, B rows 2-3, B cols 0-1 -> C cols 0-1
        # Block 4: A cols 2-3, B rows 2-3, B cols 2-3 -> C cols 2-3
        #
        # This covers all terms:
        # C[i, 0..1] = A[i, 0..1] B[0..1, 0..1] + A[i, 2..3] B[2..3, 0..1]
        # C[i, 2..3] = A[i, 0..1] B[0..1, 2..3] + A[i, 2..3] B[2..3, 2..3]
        #
        # This is correct.
        
        strassen_steps = [
            {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
            {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
            {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
            {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
            {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
        ]
        
        def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
            for s in range(7):
                step = strassen_steps[s]
                col = col_offset + s
                for local_idx, coeff in step['u'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = l_row
                    global_j = a_col_start + l_col
                    global_idx = global_i * m + global_j
                    U[global_idx][col] += coeff
                for local_idx, coeff in step['v'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = b_row_start + l_row
                    global_j = b_col_start + l_col
                    global_idx = global_i * p + global_j
                    V[global_idx][col] += coeff
                for local_idx, coeff in step['w'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_k = c_col_start + l_col
                    global_i = l_row
                    global_idx = global_k * n + global_i
                    W[global_idx][col] += coeff
        
        add_block(0, 0, 0, 0, 0)
        add_block(0, 0, 2, 2, 7)
        add_block(2, 2, 0, 0, 14)
        add_block(2, 2, 2, 2, 21)
        
        return U, V, W
    
    U, V, W = build_strassen()
    
    # Verify
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        rank = U_mat.shape[1]
        U_m = U_mat.reshape(n, m, rank)
        V_m = V_mat.reshape(m, p, rank)
        W_m = W_mat.reshape(p, n, rank)
        
        result = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_mat[:, r].reshape(n, m)
            v_col = V_mat[:, r].reshape(m, p)
            w_col = W_mat[:, r].reshape(p, n)
            result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
        
        return np.allclose(result, 1.0, atol=tol)
    
    if verify_fast(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to identity if verification fails (should not happen)
        rank = n * m * p
        U = np.eye(rank).reshape(n*m, rank)
        V = np.eye(rank).reshape(m*p, rank)
        W = np.eye(rank).reshape(p*n, rank)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
