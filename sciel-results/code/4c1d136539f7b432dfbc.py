"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Strategy:
    Decompose the 2x4 * 4x4 multiplication into four independent 2x2 * 2x2 multiplications
    by splitting the middle dimension (columns of A / rows of B) into two halves {0,1} and {2,3},
    and splitting the output dimension (columns of B) into two halves {0,1} and {2,3}.
    
    The four sub-problems are:
    1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    2. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    3. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    Each 2x2 * 2x2 multiplication is performed using the Strassen algorithm, which has rank 7.
    Since the sub-problems are disjoint in terms of the rank-1 tensor components (they share inputs 
    but the Strassen linear combinations are independent per block), the total rank is 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    
    # We will build the factors by iterating through the 4 blocks and the 7 Strassen steps.
    # Total rank = 28.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2 * 2x2
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y10 - y11)
    # M4 = x11(y01 - y00)
    # M5 = (x00 + x10)y11
    # M6 = (x11 - x00)(y00 + y10)
    # M7 = (x01 - x11)(y11 + y01)
    
    # Z00 = M1 - M4 + M5 + M6
    # Z01 = M3 + M4
    # Z10 = M2 + M5
    # Z11 = M1 - M2 + M3 + M7
    
    # Define the 7 multiplications as (coeffs_x, coeffs_y)
    # coeffs is a dict mapping local index (0,1,2,3) to coefficient
    # Local X indices: 00->0, 01->1, 10->2, 11->3
    # Local Y indices: 00->0, 01->1, 10->2, 11->3
    
    strassen_ms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}),
        # M3
        ({0: 1}, {2: 1, 3: -1}),
        # M4
        ({3: 1}, {1: 1, 0: -1}),
        # M5
        ({0: 1, 2: 1}, {3: 1}),
        # M6
        ({3: 1, 0: -1}, {0: 1, 2: 1}),
        # M7
        ({1: 1, 3: -1}, {3: 1, 1: 1})
    ]
    
    # Define the 4 output additions as linear combos of M's
    # Z00 = M1 - M4 + M5 + M6
    # Z01 = M3 + M4
    # Z10 = M2 + M5
    # Z11 = M1 - M2 + M3 + M7
    # coeffs is a dict mapping M-index (0..6) to coefficient
    z_combos = [
        {0: 1, 3: -1, 4: 1, 5: 1},  # Z00
        {2: 1, 3: 1},                # Z01
        {1: 1, 4: 1},                # Z10
        {0: 1, 1: -1, 2: 1, 6: 1}   # Z11
    ]
    
    # Define the 4 blocks
    # Block 0: A_L * B_LL -> C_LL
    # A_L: rows 0-1, cols 0-1. Global U indices: i*4 + j. i in 0,1; j in 0,1.
    # B_LL: rows 0-1, cols 0-1. Global V indices: j*4 + k. j in 0,1; k in 0,1.
    # C_LL: rows 0-1, cols 0-1. Global W indices: k*2 + i. k in 0,1; i in 0,1.
    
    # Block 1: A_L * B_LR -> C_LR
    # A_L: same.
    # B_LR: rows 0-1, cols 2-3. Global V indices: j*4 + k. j in 0,1; k in 2,3.
    # C_LR: rows 0-1, cols 2-3. Global W indices: k*2 + i. k in 2,3; i in 0,1.
    
    # Block 2: A_R * B_RL -> C_RL
    # A_R: rows 0-1, cols 2-3. Global U indices: i*4 + j. i in 0,1; j in 2,3.
    # B_RL: rows 2-3, cols 0-1. Global V indices: j*4 + k. j in 2,3; k in 0,1.
    # C_RL: rows 0-1, cols 0-1. Global W indices: k*2 + i. k in 0,1; i in 0,1.
    
    # Block 3: A_R * B_RR -> C_RR
    # A_R: same.
    # B_RR: rows 2-3, cols 2-3. Global V indices: j*4 + k. j in 2,3; k in 2,3.
    # C_RR: rows 0-1, cols 2-3. Global W indices: k*2 + i. k in 2,3; i in 0,1.
    
    def get_global_u(i_local, j_local, block_a_offset):
        # i_local in {0,1}, j_local in {0,1}
        # block_a_offset is 0 for A_L (cols 0,1) or 2 for A_R (cols 2,3)
        j_global = j_local + block_a_offset
        return i_local * 4 + j_global
        
    def get_global_v(j_local, k_local, block_b_row_offset, block_b_col_offset):
        # j_local in {0,1}, k_local in {0,1}
        # block_b_row_offset is 0 for rows 0,1 or 2 for rows 2,3
        # block_b_col_offset is 0 for cols 0,1 or 2 for cols 2,3
        j_global = j_local + block_b_row_offset
        k_global = k_local + block_b_col_offset
        return j_global * 4 + k_global
        
    def get_global_w(i_local, k_local, block_c_col_offset):
        # i_local in {0,1}, k_local in {0,1}
        # block_c_col_offset is 0 for cols 0,1 or 2 for cols 2,3
        k_global = k_local + block_c_col_offset
        # W index is k*n + i = k*2 + i
        return k_global * 2 + i_local
        
    # Process each block
    block_configs = [
        # (a_offset, b_row_offset, b_col_offset, c_col_offset)
        (0, 0, 0, 0), # Block 0: A_L, B_LL, C_LL
        (0, 0, 2, 2), # Block 1: A_L, B_LR, C_LR
        (2, 2, 0, 0), # Block 2: A_R, B_RL, C_RL
        (2, 2, 2, 2)  # Block 3: A_R, B_RR, C_RR
    ]
    
    rank_idx = 0
    for a_off, b_row_off, b_col_off, c_col_off in block_configs:
        for m_idx in range(7):
            cx, cy = strassen_ms[m_idx]
            # For each output z in this block, add contribution to W
            # But we need to construct U, V, W for this specific rank term (m_idx)
            
            # U and V are determined by the Strassen multiplication m_idx
            # W is determined by the sum of these m's for each output z
            
            # However, the decomposition is a sum of rank-1 tensors.
            # Each rank-1 tensor corresponds to one (block, m_idx) pair?
            # No, Strassen M's are intermediate. The rank-1 tensors in the tensor decomposition
            # correspond to the bilinear forms M1..M7.
            # The output Z is a linear combination of M's.
            # So, for each block and each M, we have one rank-1 tensor?
            # Let's check.
            # T = sum_{block} sum_{m} (sum_{z} coeff_z_m * z_term) * x_term_m * y_term_m
            # This can be rewritten as sum_{block} sum_{m} (x_term_m * y_term_m * (sum_{z} coeff_z_m * z_term))
            # This is a sum of rank-1 tensors?
            # A rank-1 tensor is u * v * w.
            # Here, for a fixed block and m, the term is:
            # (linear combo of X) * (linear combo of Y) * (linear combo of Z)
            # This is indeed a rank-1 tensor in the tensor product space.
            # So yes, we have 4 blocks * 7 M's = 28 rank-1 tensors.
            
            # Construct U for this term
            for x_loc, coeff in cx.items():
                u_idx = get_global_u(0, x_loc, a_off) if x_loc < 2 else get_global_u(1, x_loc - 2, a_off)
                # Wait, x_loc is 0,1,2,3 corresponding to 00, 01, 10, 11
                # i_local = x_loc // 2 ? No.
                # 0->00 (i=0, j=0), 1->01 (i=0, j=1), 2->10 (i=1, j=0), 3->11 (i=1, j=1)
                i_loc = 0 if x_loc < 2 else 1
                j_loc = x_loc % 2
                u_idx = get_global_u(i_loc, j_loc, a_off)
                U[u_idx][rank_idx] += coeff
                
            for y_loc, coeff in cy.items():
                # y_loc 0->00, 1->01, 2->10, 3->11
                # j_loc = y_loc // 2 ? No.
                # 0->00 (j=0, k=0), 1->01 (j=0, k=1), 2->10 (j=1, k=0), 3->11 (j=1, k=1)
                j_loc = 0 if y_loc < 2 else 1
                k_loc = y_loc % 2
                v_idx = get_global_v(j_loc, k_loc, b_row_off, b_col_off)
                V[v_idx][rank_idx] += coeff
                
            # Construct W for this term
            # We need to add the contribution of this M to the appropriate Z's.
            # The term in the tensor is M * (sum_{z} coeff_z_m * e_z)
            # So W[z_idx][rank_idx] += coeff_z_m
            for z_loc, coeff in z_combos[0].items(): # Z00
                if m_idx == z_loc: # This logic is wrong. z_combos maps z to m's.
                    pass
            
            # Correct logic for W:
            # For each z (0..3), find the coefficient of this m (m_idx) in z's formula.
            # z_combos[z] is a dict {m_idx: coeff}
            for z_loc in range(4):
                coeff = z_combos[z_loc].get(m_idx, 0)
                if coeff != 0:
                    # z_loc 0->00, 1->01, 2->10, 3->11
                    # i_loc = z_loc // 2 ? No.
                    # 0->00 (i=0, k=0), 1->01 (i=0, k=1), 2->10 (i=1, k=0), 3->11 (i=1, k=1)
                    i_loc = 0 if z_loc < 2 else 1
                    k_loc = z_loc % 2
                    w_idx = get_global_w(i_loc, k_loc, c_col_off)
                    W[w_idx][rank_idx] += coeff
                    
            rank_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
