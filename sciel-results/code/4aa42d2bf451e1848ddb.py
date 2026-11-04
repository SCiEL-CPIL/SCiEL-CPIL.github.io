import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Strassen decomposition for 2x2x2
    # U_s, V_s, W_s are 4x7 matrices
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # Term 0 (M1): (A00+A11)(B00+B11)
    U_s[0, 0] += 1.0
    U_s[3, 0] += 1.0
    V_s[0, 0] += 1.0
    V_s[3, 0] += 1.0
    W_s[0, 0] += 1.0
    W_s[3, 0] += 1.0
    
    # Term 1 (M2): (A10+A11)B00
    U_s[2, 1] += 1.0
    U_s[3, 1] += 1.0
    V_s[0, 1] += 1.0
    W_s[2, 1] += 1.0
    W_s[3, 1] -= 1.0
    
    # Term 2 (M3): A00(B01-B11)
    U_s[0, 2] += 1.0
    V_s[1, 2] += 1.0
    V_s[3, 2] -= 1.0
    W_s[1, 2] += 1.0
    W_s[3, 2] += 1.0
    
    # Term 3 (M4): A11(B10-B00)
    U_s[3, 3] += 1.0
    V_s[2, 3] += 1.0
    V_s[0, 3] -= 1.0
    W_s[0, 3] += 1.0
    W_s[2, 3] += 1.0
    
    # Term 4 (M5): (A00+A01)B11
    U_s[0, 4] += 1.0
    U_s[1, 4] += 1.0
    V_s[3, 4] += 1.0
    W_s[0, 4] -= 1.0
    W_s[1, 4] += 1.0
    
    # Term 5 (M6): (A10-A00)(B00+B01)
    U_s[2, 5] += 1.0
    U_s[0, 5] -= 1.0
    V_s[0, 5] += 1.0
    V_s[1, 5] += 1.0
    W_s[3, 5] += 1.0
    
    # Term 6 (M7): (A01-A11)(B10+B11)
    U_s[1, 6] += 1.0
    U_s[3, 6] -= 1.0
    V_s[2, 6] += 1.0
    V_s[3, 6] += 1.0
    W_s[0, 6] += 1.0

    # Construct Rank 42 decomposition
    rank_42 = 42
    U42 = np.zeros((nm, rank_42), dtype=float)
    V42 = np.zeros((mp, rank_42), dtype=float)
    W42 = np.zeros((pn, rank_42), dtype=float)
    
    # 6 blocks: 3 Top (A_L), 3 Bottom (A_R)
    # Block 0: A_L, B_T1, C_0
    # Block 1: A_L, B_T2, C_1
    # Block 2: A_L, B_T3, C_2
    # Block 3: A_R, B_B1, C_0
    # Block 4: A_R, B_B2, C_1
    # Block 5: A_R, B_B3, C_2
    
    for block_id in range(6):
        # Determine A offset (0 for Top/L, 2 for Bottom/R)
        a_col_offset = 0 if block_id < 3 else 2
        # Determine B row offset (0 for Top, 2 for Bottom)
        b_row_offset = 0 if block_id < 3 else 2
        # Determine B col offset (0, 2, 4)
        b_col_offset = (block_id % 3) * 2
        # Determine C col offset (0, 2, 4)
        c_col_offset = (block_id % 3) * 2
        
        base_rank = block_id * 7
        
        for r_s in range(7):
            r_total = base_rank + r_s
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    local_a_idx = i * 2 + j
                    val_s = U_s[local_a_idx, r_s]
                    if val_s != 0:
                        global_a_idx = i * 4 + (j + a_col_offset)
                        U42[global_a_idx, r_total] += val_s
                        
            # Fill V
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    val_s = V_s[local_b_idx, r_s]
                    if val_s != 0:
                        global_b_idx = (r + b_row_offset) * 6 + (c + b_col_offset)
                        V42[global_b_idx, r_total] += val_s
                        
            # Fill W
            for k in range(2):
                for i in range(2):
                    local_c_idx = k * 2 + i
                    val_s = W_s[local_c_idx, r_s]
                    if val_s != 0:
                        # C index is k*n + i. Here n=2.
                        # Global C column is c_col_offset + i?
                        # Wait, C is 2x6.
                        # C[k, c] where c is 0..5.
                        # c = c_col_offset + i?
                        # No, i is the row index of C (0..1).
                        # The local C is 2x2.
                        # Local C index lc = k*2 + i.
                        # Global C index gc = k*2 + i?
                        # No.
                        # Global C is 2x6.
                        # The block C_block is 2x2.
                        # It occupies columns c_col_offset and c_col_offset+1 of Global C.
                        # So Global C row is k (0..1).
                        # Global C col is c_col_offset + i?
                        # No, i is the local row index (0..1).
                        # The local C is C_local[k, i].
                        # This maps to Global C[k, c_col_offset + i].
                        # Flattened Global C index = k * 6 + (c_col_offset + i)?
                        # No, the task says W[k*n+i, r].
                        # n=2. So W index is k*2 + i.
                        # This implies the output matrix is stored in column-major order?
                        # "W[k*n+i,r]"
                        # If n=2, indices are 0,1 for k=0; 2,3 for k=1.
                        # This corresponds to C[i, k] if C is 2x6?
                        # Let's check the tensor definition.
                        # T[i, j, k] = delta.
                        # C = A B.
                        # C[i, k] = sum_j A[i, j] B[j, k].
                        # The output factor W has shape (p*n, rank) = (12, rank).
                        # Index is k*n + i.
                        # So W index corresponds to (k, i).
                        # In the local Strassen block, we have C_local[k, i].
                        # So the local index lc = k*2 + i matches the global index structure for the block.
                        # However, the global index must account for the block position.
                        # The block is at columns c_col_offset and c_col_offset+1.
                        # So Global k is still k (0..1).
                        # Global i is i (0..1).
                        # BUT, the global index is k*n + i.
                        # This index is fixed for the position (k, i).
                        # Wait. If we have multiple blocks writing to the same (k, i) position, we sum them.
                        # Block 0 writes to C[k, 0..1].
                        # Block 3 writes to C[k, 0..1].
                        # So for Block 0, Global C index is k*2 + i.
                        # For Block 3, Global C index is k*2 + i.
                        # They add up.
                        # This is correct.
                        
                        global_c_idx = k * 2 + i
                        W42[global_c_idx, r_total] += val_s

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(len(U_mat[0])):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if verify(U42, V42, W42):
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    else:
        # Fallback to Rank 48
        return construct_rank_48()

def construct_rank_48():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 48
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
