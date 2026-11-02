import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # Try rank 40: 4 blocks of 3x2 x 2x2 -> 3x2, each rank 10
    # Construct rank-10 decomposition for 3x2 x 2x2 -> 3x2
    
    def build_3x2x2_rank10():
        """
        A: 3x2 (indices 0-5: row*2+col)
        B: 2x2 (indices 0-3: row*2+col)
        C: 3x2 (indices 0-5: row*2+col)
        
        C[i,j] = sum_k A[i,k] * B[k,j]
        """
        U = np.zeros((6, 10))
        V = np.zeros((4, 10))
        W = np.zeros((6, 10))
        
        # Strassen for rows 0,1 of A (7 products)
        # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00 -> contributes to C10, C11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00*(B10-B00) -> contributes to C01, C11
        U[0, 2] = 1
        V[2, 2] = 1; V[0, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B01-B11) -> contributes to C00, C10
        U[3, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11 -> contributes to C00, C01
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01) -> contributes to C11
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11) -> contributes to C00
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # Now for row 2: C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        #             C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        # Need 4 more products, but maybe we can do it in 3?
        # Try: M8 = A[2,0]*(B[0,0]+B[1,0]), M9 = A[2,1]*(B[0,1]+B[1,1]), M10 = ?
        # That doesn't directly work.
        
        # Let's just use 4 trivial products for now (rank 11) and see if we can optimize
        # Actually, let's try a different Strassen variant that might help
        
        # For now, return rank 11 version
        return None
    
    # Since rank-10 for 3x2x2 is non-trivial, let's stick with rank 44 but verify correctness
    # and add a fallback mechanism
    
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
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
    
    # Extra part (cols 7-10) for Row 2 of A
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Mapping function
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
                
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
                
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify decomposition is correct
    if time.time() < deadline:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    a_row, a_col = divmod(a, m)
                    b_row, b_col = divmod(b, p)
                    c_col, c_row = divmod(c, n)
                    
                    expected = 1.0 if (a_col == b_row and c_row == a_row and c_col == b_col) else 0.0
                    computed = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    
                    if abs(computed - expected) > 1e-10:
                        # Fallback to trivial rank-48
                        U_fallback = np.zeros((n*m, n*m*p))
                        V_fallback = np.zeros((m*p, n*m*p))
                        W_fallback = np.zeros((p*n, n*m*p))
                        for a_idx in range(n*m):
                            a_r, a_c = divmod(a_idx, m)
                            for b_idx in range(m*p):
                                b_r, b_c = divmod(b_idx, p)
                                if a_c == b_r:
                                    c_idx = b_c * n + a_r
                                    rank_idx = a_idx * (m*p) + b_idx
                                    U_fallback[a_idx, rank_idx] = 1
                                    V_fallback[b_idx, rank_idx] = 1
                                    W_fallback[c_idx, rank_idx] = 1
                        return {
                            "U": U_fallback.tolist(),
                            "V": V_fallback.tolist(),
                            "W": W_fallback.tolist()
                        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
