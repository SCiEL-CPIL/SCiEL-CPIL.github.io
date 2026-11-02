import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def get_current_solution():
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10)
        U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
        U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
        U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the solution is correct
    def verify_solution(sol, max_tests=10):
        U = np.array(sol['U'])
        V = np.array(sol['V'])
        W = np.array(sol['W'])
        rank = U.shape[1]
        
        # The tensor condition: for all i (A index), j (B index), k (C index):
        # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
        # where T[i,j,k] = 1 if A[i] and B[j] contribute to C[k] in the multiplication, else 0.
        # A index i = row_a * 4 + col_a (0..11)
        # B index j = row_b * 4 + col_b (0..15)
        # C index k = col_c * 3 + row_c (0..11) [trace convention W[k*n+i,r]]
        
        # C[row_c, col_c] = sum_{j} A[row_a, col_a] * B[col_a, col_c]
        # So T[i, j, k] = 1 if:
        #   i = row_a * 4 + col_a
        #   j = col_a * 4 + col_c  (B index: row_B=col_a, col_B=col_c)
        #   k = col_c * 3 + row_c  (C index per trace convention)
        #   and row_a is the row of A, col_c is the col of C
        
        # Let's build T
        T = np.zeros((12, 16, 12))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    T[i, j, k] = 1.0
        
        # Check: sum_r U[:,r] outer V[:,r] outer W[:,r] == T
        # For efficiency, check on random samples
        for _ in range(max_tests):
            i = np.random.randint(0, 12)
            j = np.random.randint(0, 16)
            k = np.random.randint(0, 12)
            computed = np.sum(U[i, :] * V[j, :] * W[k, :])
            expected = T[i, j, k]
            if abs(computed - expected) > 1e-9:
                return False
        return True
    
    sol = get_current_solution()
    
    # Verify correctness
    np.random.seed(42)
    if not verify_solution(sol, max_tests=100):
        # If verification fails, return trivial solution as fallback
        rank = 48
        U = np.zeros((12, rank))
        V = np.zeros((16, rank))
        W = np.zeros((12, rank))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    idx = (row_a * 4 + col_a) * 4 + col_c
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    U[i, idx] = 1
                    V[j, idx] = 1
                    W[k, idx] = 1
        sol = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check deadline
    if time.time() > deadline:
        return sol
    
    return sol
