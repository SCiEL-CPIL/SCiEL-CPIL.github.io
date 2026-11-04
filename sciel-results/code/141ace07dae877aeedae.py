import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = 3, 4, 7
    # Target rank is 78
    target_rank = 78
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    current_rank = 0
    
    def add_block(A_idx, B_idx, C_idx, block_type):
        nonlocal current_rank
        # A_idx: (i_start, i_end, j_start, j_end) for A[i,j]
        # B_idx: (j_start, j_end, k_start, k_end) for B[j,k]
        # C_idx: (i_start, i_end, k_start, k_end) for C[i,k]
        
        i_s, i_e, j_s, j_e = A_idx
        j_s2, j_e2, k_s, k_e = B_idx
        # Note: j_s2 should match j_s and j_e2 should match j_e for valid multiplication
        if j_s2 != j_s or j_e2 != j_e:
            raise ValueError("Index mismatch")
        
        if block_type == "1x4x1":
            # A is 1x4, B is 4x1, C is 1x1
            # Rank 4
            for j in range(j_s, j_e):
                U[i_s * m + j, current_rank] = 1
                V[j * p + k_s, current_rank] = 1
                W[k_s * n + i_s, current_rank] = 1
                current_rank += 1
                
        elif block_type == "1x4x2":
            # A is 1x4, B is 4x2, C is 1x2
            # Rank 8
            # Split B into two 4x1 blocks
            for k in range(k_s, k_e):
                for j in range(j_s, j_e):
                    U[i_s * m + j, current_rank] = 1
                    V[j * p + k, current_rank] = 1
                    W[k * n + i_s, current_rank] = 1
                    current_rank += 1
                    
        elif block_type == "2x4x1":
            # A is 2x4, B is 4x1, C is 2x1
            # Rank 8
            for i in range(i_s, i_e):
                for j in range(j_s, j_e):
                    U[i * m + j, current_rank] = 1
                    V[j * p + k_s, current_rank] = 1
                    W[k_s * n + i, current_rank] = 1
                    current_rank += 1
                    
        elif block_type == "2x4x2":
            # A is 2x4, B is 4x2, C is 2x2
            # Rank 14 (two Strassen 2x2x2)
            # Split A into A1 (2x2), A2 (2x2)
            # Split B into B1 (2x2), B2 (2x2)
            # C = A1 B1 + A2 B2
            
            # Strassen for A1 B1
            # A1 is rows i_s..i_e, cols j_s..j_s+2
            # B1 is rows j_s..j_s+2, cols k_s..k_e
            # We need a Strassen helper
            def strassen_2x2x2(a_i_s, a_j_s, b_j_s, b_k_s, c_i_s, c_k_s):
                nonlocal current_rank
                # A is 2x2 at (a_i_s, a_j_s)
                # B is 2x2 at (b_j_s, b_k_s)
                # C is 2x2 at (c_i_s, c_k_s)
                
                # M1 = (A11 + A22)(B21 + B22)
                # M2 = (A21 + A22)B11
                # M3 = A11(B12 - B22)
                # M4 = A22(B21 - B11)
                # M5 = (A11 + A12)B22
                # M6 = (A21 - A11)(B11 + B12)
                # M7 = (A12 - A22)(B21 + B22)
                
                # Indices for A:
                # A11: (a_i_s, a_j_s)
                # A12: (a_i_s, a_j_s+1)
                # A21: (a_i_s+1, a_j_s)
                # A22: (a_i_s+1, a_j_s+1)
                
                # Indices for B:
                # B11: (b_j_s, b_k_s)
                # B12: (b_j_s, b_k_s+1)
                # B21: (b_j_s+1, b_k_s)
                # B22: (b_j_s+1, b_k_s+1)
                
                # Indices for C:
                # C11: (c_i_s, c_k_s)
                # C12: (c_i_s, c_k_s+1)
                # C21: (c_i_s+1, c_k_s)
                # C22: (c_i_s+1, c_k_s+1)
                
                def add_term(u_val, u_idx, v_val, v_idx, w_val, w_idx):
                    nonlocal current_rank
                    U[u_idx, current_rank] += u_val
                    V[v_idx, current_rank] += v_val
                    W[w_idx, current_rank] += w_val
                    current_rank += 1

                # M1
                # (A11 + A22)(B21 + B22)
                # A part: A11, A22
                # B part: B21, B22
                # C part: C12, C22, C21? No.
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                # M1 contributes to C11 (+1), C22 (+1)
                # Terms for M1:
                # (A11 + A22) * (B21 + B22)
                # = A11 B21 + A11 B22 + A22 B21 + A22 B22
                # This is 4 terms in schoolbook, but Strassen combines them.
                # Wait, Strassen M1 is a single multiplication of sums.
                # U column: A11 + A22
                # V column: B21 + B22
                # W column: C11 + C22? No.
                # The decomposition is:
                # M1 = (A11 + A22)(B21 + B22)
                # This product is added to C11 and C22?
                # No, C11 = M1 + M4 - M5 + M7
                # C22 = M1 - M2 + M3 + M6
                # So M1 contributes +1 to C11 and +1 to C22.
                
                # We add 1 column for M1.
                # U: A11, A22 are 1
                # V: B21, B22 are 1
                # W: C11, C22 are 1
                
                u_idx_a11 = (a_i_s) * m + (a_j_s)
                u_idx_a22 = (a_i_s+1) * m + (a_j_s+1)
                v_idx_b21 = (b_j_s+1) * p + (b_k_s)
                v_idx_b22 = (b_j_s+1) * p + (b_k_s+1)
                w_idx_c11 = (b_k_s) * n + (a_i_s)
                w_idx_c22 = (b_k_s+1) * n + (a_i_s+1)
                
                U[u_idx_a11, current_rank] = 1
                U[u_idx_a22, current_rank] = 1
                V[v_idx_b21, current_rank] = 1
                V[v_idx_b22, current_rank] = 1
                W[w_idx_c11, current_rank] = 1
                W[w_idx_c22, current_rank] = 1
                current_rank += 1
                
                # M2 = (A21 + A22)B11
                # Contributes +1 to C21, -1 to C22
                u_idx_a21 = (a_i_s+1) * m + (a_j_s)
                v_idx_b11 = (b_j_s) * p + (b_k_s)
                w_idx_c21 = (b_k_s) * n + (a_i_s+1)
                
                U[u_idx_a21, current_rank] = 1
                U[u_idx_a22, current_rank] = 1
                V[v_idx_b11, current_rank] = 1
                W[w_idx_c21, current_rank] = 1
                W[w_idx_c22, current_rank] = -1
                current_rank += 1
                
                # M3 = A11(B12 - B22)
                # Contributes +1 to C12, +1 to C22
                v_idx_b12 = (b_j_s) * p + (b_k_s+1)
                w_idx_c12 = (b_k_s+1) * n + (a_i_s)
                
                U[u_idx_a11, current_rank] = 1
                V[v_idx_b12, current_rank] = 1
                V[v_idx_b22, current_rank] = -1
                W[w_idx_c12, current_rank] = 1
                W[w_idx_c22, current_rank] = 1
                current_rank += 1
                
                # M4 = A22(B21 - B11)
                # Contributes +1 to C11, +1 to C21
                U[u_idx_a22, current_rank] = 1
                V[v_idx_b21, current_rank] = 1
                V[v_idx_b11, current_rank] = -1
                W[w_idx_c11, current_rank] = 1
                W[w_idx_c21, current_rank] = 1
                current_rank += 1
                
                # M5 = (A11 + A12)B22
                # Contributes -1 to C11, +1 to C12
                u_idx_a12 = (a_i_s) * m + (a_j_s+1)
                U[u_idx_a11, current_rank] = 1
                U[u_idx_a12, current_rank] = 1
                V[v_idx_b22, current_rank] = 1
                W[w_idx_c11, current_rank] = -1
                W[w_idx_c12, current_rank] = 1
                current_rank += 1
                
                # M6 = (A21 - A11)(B11 + B12)
                # Contributes +1 to C22
                U[u_idx_a21, current_rank] = 1
                U[u_idx_a11, current_rank] = -1
                V[v_idx_b11, current_rank] = 1
                V[v_idx_b12, current_rank] = 1
                W[w_idx_c22, current_rank] = 1
                current_rank += 1
                
                # M7 = (A12 - A22)(B21 + B22)
                # Contributes +1 to C11
                U[u_idx_a12, current_rank] = 1
                U[u_idx_a22, current_rank] = -1
                V[v_idx_b21, current_rank] = 1
                V[v_idx_b22, current_rank] = 1
                W[w_idx_c11, current_rank] = 1
                current_rank += 1

            # Call Strassen for A1 B1
            # A1 is rows i_s..i_e, cols j_s..j_s+2
            # B1 is rows j_s..j_s+2, cols k_s..k_e
            strassen_2x2x2(i_s, j_s, j_s, k_s, i_s, k_s)
            
            # Call Strassen for A2 B2
            # A2 is rows i_s..i_e, cols j_s+2..j_e
            # B2 is rows j_s+2..j_e, cols k_s..k_e
            strassen_2x2x2(i_s, j_s+2, j_s+2, k_s, i_s, k_s)

    # Block 1: A1 B1a (1x4x1)
    # A1: row 0, cols 0-3. B1a: rows 0-3, col 0.
    add_block((0, 1, 0, 4), (0, 4, 0, 1), None, "1x4x1")
    
    # Block 2: A1 B1b (1x4x2)
    # A1: row 0, cols 0-3. B1b: rows 0-3, cols 1-2.
    add_block((0, 1, 0, 4), (0, 4, 1, 3), None, "1x4x2")
    
    # Block 3: A1 B2a (1x4x2)
    # A1: row 0, cols 0-3. B2a: rows 0-3, cols 3-4.
    add_block((0, 1, 0, 4), (0, 4, 3, 5), None, "1x4x2")
    
    # Block 4: A1 B2b (1x4x2)
    # A1: row 0, cols 0-3. B2b: rows 0-3, cols 5-6.
    add_block((0, 1, 0, 4), (0, 4, 5, 7), None, "1x4x2")
    
    # Block 5: A2 B1a (2x4x1)
    # A2: rows 1-2, cols 0-3. B1a: rows 0-3, col 0.
    add_block((1, 3, 0, 4), (0, 4, 0, 1), None, "2x4x1")
    
    # Block 6: A2 B1b (2x4x2)
    # A2: rows 1-2, cols 0-3. B1b: rows 0-3, cols 1-2.
    add_block((1, 3, 0, 4), (0, 4, 1, 3), None, "2x4x2")
    
    # Block 7: A2 B2a (2x4x2)
    # A2: rows 1-2, cols 0-3. B2a: rows 0-3, cols 3-4.
    add_block((1, 3, 0, 4), (0, 4, 3, 5), None, "2x4x2")
    
    # Block 8: A2 B2b (2x4x2)
    # A2: rows 1-2, cols 0-3. B2b: rows 0-3, cols 5-6.
    add_block((1, 3, 0, 4), (0, 4, 5, 7), None, "2x4x2")
    
    if current_rank != target_rank:
        raise ValueError(f"Rank mismatch: expected {target_rank}, got {current_rank}")
        
    return {"U": U, "V": V, "W": W}
