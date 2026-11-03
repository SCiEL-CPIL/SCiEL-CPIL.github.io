import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 + Rank 16 for 4x4x1 = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 65
    V = [[0] * rank for _ in range(m * p)] # 20 x 65
    W = [[0] * rank for _ in range(p * n)] # 20 x 65
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # We define the 7 block products M1..M7
    # Each M is a 2x2 matrix product.
    # We iterate through the 7 blocks and for each, iterate through the 7 Strassen terms.
    
    # Helper to get indices for a 2x2 block in a matrix
    # A is 4x4, flattened row-major.
    # Block (br, bc) of size 2x2.
    # br, bc in {0, 1}
    
    def get_A_indices(br, bc):
        # br: 0 or 1, bc: 0 or 1
        rows = [br*2, br*2+1]
        cols = [bc*2, bc*2+1]
        return [r*4 + c for r in rows for c in cols]

    def get_B_indices(br, bc):
        # B is 4x5. We use cols 0-3 for the 4x4 part.
        rows = [br*2, br*2+1]
        cols = [bc*2, bc*2+1]
        return [r*5 + c for r in rows for c in cols]

    def get_C_indices(br, bc):
        # C is 4x5. We use cols 0-3 for the 4x4 part.
        # W index is k*n + i. k is col, i is row.
        # Block (br, bc) of C (4x4 part).
        # br corresponds to row block of C?
        # C = A B. A is 4x4, B is 4x4.
        # C is 4x4.
        # C block (br, bc) is rows br*2..br*2+1, cols bc*2..bc*2+1.
        # W index: k (col) * 4 + i (row).
        rows = [br*2, br*2+1]
        cols = [bc*2, bc*2+1]
        return [k*4 + i for k in cols for i in rows]

    # Define the 7 Strassen terms for 2x2
    # Each term is (u_vec, v_vec, w_vec) for 2x2 indices
    # u_vec corresponds to A 2x2 indices [0,1,2,3] -> [00, 01, 10, 11]
    # v_vec corresponds to B 2x2 indices [0,1,2,3] -> [00, 01, 10, 11]
    # w_vec corresponds to C 2x2 indices [0,1,2,3] -> [00, 01, 10, 11]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Define the 7 block products for 4x4
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each block product, we need to map the 2x2 Strassen terms to the 4x4 indices.
    # The inputs to the 2x2 Strassen are linear combinations of the 4x4 blocks.
    
    block_products = [
        # (A_coeffs, B_coeffs, C_block_indices)
        # A_coeffs: dict of block -> coeff
        # B_coeffs: dict of block -> coeff
        # C_block: (br, bc) of the result block? No, the result is distributed.
        # Actually, each M_t is a full 2x2 matrix.
        # So the C indices for the 2x2 Strassen terms map to specific 2x2 blocks of C.
        # M1 contributes to C11 and C22? No.
        # M1 is a 2x2 matrix. It contributes to all 4 blocks of C?
        # No, M1 is the product of (A11+A22) and (B11+B22).
        # This is a 2x2 matrix.
        # How does it map to C?
        # C = A B.
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        
        # Strassen for 4x4:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C11 and C22.
        # M2 contributes to C21 and C22.
        # M3 contributes to C12 and C22.
        # M4 contributes to C11 and C21.
        # M5 contributes to C11 and C12.
        # M6 contributes to C22.
        # M7 contributes to C11.
        
        # This means for each M_t, we need to expand its 2x2 Strassen terms.
        # The 2x2 Strassen terms produce a 2x2 result.
        # But M_t is just one term in the sum for C_blocks.
        # Wait, the 2x2 Strassen decomposition computes the full 2x2 product.
        # So M1 is a 2x2 matrix.
        # The elements of M1 are added to specific elements of C.
        # Specifically, M1 (as a 2x2 matrix) is added to C11 (top-left 2x2) and C22 (bottom-right 2x2).
        # So for M1, the C indices are the union of C11 and C22 indices.
        # But the 2x2 Strassen terms for M1 produce 4 output values (the 2x2 matrix M1).
        # Let's call them M1_00, M1_01, M1_10, M1_11.
        # M1_00 is added to C_00 (global index 0).
        # M1_01 is added to C_02 (global index 2).
        # M1_10 is added to C_20 (global index 8).
        # M1_11 is added to C_22 (global index 10).
        
        # So for M1, the C mapping is:
        # Local 00 -> Global C[0,0]
        # Local 01 -> Global C[0,2]
        # Local 10 -> Global C[2,0]
        # Local 11 -> Global C[2,2]
        
        # Similarly for M2:
        # M2 contributes to C21 and C22.
        # M2_00 -> C[2,0]
        # M2_01 -> C[2,2]
        # M2_10 -> C[3,0]
        # M2_11 -> C[3,2]
        
        # This logic holds for all M_t.
    ]
    
    # Let's define the mapping for each M_t
    # M_t is defined by A_linear_comb and B_linear_comb.
    # A_linear_comb: list of (block, coeff)
    # B_linear_comb: list of (block, coeff)
    # C_mapping: list of (local_c_idx, global_c_idx)
    
    m_definitions = [
        # M1
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [(0, 0), (1, 2), (2, 8), (3, 10)] # Local 0,1,2,3 -> Global
        },
        # M2
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [(0, 8), (1, 10), (2, 12), (3, 14)]
        },
        # M3
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [(0, 2), (1, 4), (2, 10), (3, 12)]
        },
        # M4
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [(0, 0), (1, 2), (2, 8), (3, 10)]
        },
        # M5
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [(0, 0), (1, 2), (2, 4), (3, 6)]
        },
        # M6
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [(0, 10), (1, 12), (2, 14), (3, 16)]
        },
        # M7
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [(0, 0), (1, 2), (2, 4), (3, 6)]
        }
    ]
    
    # Note: C indices for M6 and M7 need to be checked.
    # C22 = M1 - M2 + M3 + M6.
    # M6 contributes to C22.
    # M6 is 2x2.
    # M6_00 -> C[2,2] (10)
    # M6_01 -> C[2,3] (11)
    # M6_10 -> C[3,2] (14)
    # M6_11 -> C[3,3] (15)
    # My previous mapping for M6 was wrong.
    
    # Let's re-verify C mappings.
    # C is 4x4.
    # C11 (0,0..1,1) gets M1, M4, M5, M7.
    # C12 (0,2..1,3) gets M3, M5.
    # C21 (2,0..3,1) gets M2, M4.
    # C22 (2,2..3,3) gets M1, M2, M3, M6.
    
    # M1 (2x2):
    # M1_00 -> C[0,0] (0)
    # M1_01 -> C[0,2] (2)
    # M1_10 -> C[2,0] (8)
    # M1_11 -> C[2,2] (10)
    # Correct.
    
    # M2 (2x2):
    # M2_00 -> C[2,0] (8)
    # M2_01 -> C[2,2] (10)
    # M2_10 -> C[3,0] (12)
    # M2_11 -> C[3,2] (14)
    # Correct.
    
    # M3 (2x2):
    # M3_00 -> C[0,2] (2)
    # M3_01 -> C[0,3] (3)
    # M3_10 -> C[2,2] (10)
    # M3_11 -> C[2,3] (11)
    # Correct.
    
    # M4 (2x2):
    # M4_00 -> C[0,0] (0)
    # M4_01 -> C[0,2] (2)
    # M4_10 -> C[2,0] (8)
    # M4_11 -> C[2,2] (10)
    # Correct.
    
    # M5 (2x2):
    # M5_00 -> C[0,0] (0)
    # M5_01 -> C[0,2] (2)
    # M5_10 -> C[1,0] (4)
    # M5_11 -> C[1,2] (6)
    # Correct.
    
    # M6 (2x2):
    # M6_00 -> C[2,2] (10)
    # M6_01 -> C[2,3] (11)
    # M6_10 -> C[3,2] (14)
    # M6_11 -> C[3,3] (15)
    # Correct.
    
    # M7 (2x2):
    # M7_00 -> C[0,0] (0)
    # M7_01 -> C[0,2] (2)
    # M7_10 -> C[1,0] (4)
    # M7_11 -> C[1,2] (6)
    # Correct.
    
    # Update m_definitions C mappings
    m_definitions[2]['C'] = [(0, 2), (1, 3), (2, 10), (3, 11)]
    m_definitions[4]['C'] = [(0, 0), (1, 2), (2, 4), (3, 6)]
    m_definitions[5]['C'] = [(0, 10), (1, 11), (2, 14), (3, 15)]
    m_definitions[6]['C'] = [(0, 0), (1, 2), (2, 4), (3, 6)]

    col = 0
    for m_def in m_definitions:
        for u_2x2, v_2x2, w_2x2 in strassen_terms:
            # Construct U vector
            u_vec = [0] * (n*m)
            for block, coeff in m_def['A']:
                a_indices = get_A_indices(block[0], block[1])
                for idx, a_idx in enumerate(a_indices):
                    u_vec[a_idx] += coeff * u_2x2[idx]
            
            # Construct V vector
            v_vec = [0] * (m*p)
            for block, coeff in m_def['B']:
                b_indices = get_B_indices(block[0], block[1])
                for idx, b_idx in enumerate(b_indices):
                    v_vec[b_idx] += coeff * v_2x2[idx]
            
            # Construct W vector
            w_vec = [0] * (p*n)
            for local_idx, global_idx in m_def['C']:
                w_vec[global_idx] += w_2x2[local_idx]
            
            # Fill matrices
            for i in range(n*m):
                U[i][col] = u_vec[i]
            for j in range(m*p):
                V[j][col] = v_vec[j]
            for k in range(p*n):
                W[k][col] = w_vec[k]
            
            col += 1
            
    # --- Part 2: 4x4x1 Schoolbook (Rank 16) ---
    # A is 4x4, B'' is 4x1 (col 4 of B)
    # C'' is 4x1 (col 4 of C)
    # C[i, 4] = sum_j A[i, j] B''[j, 4]
    
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1 # B'' index is j*5 + 4
            W[4*n + i][col] = 1 # C'' index is 4*4 + i
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
