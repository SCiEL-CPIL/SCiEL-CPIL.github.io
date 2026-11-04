"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    # Target rank 77
    # 49 from 4x4x4 (Strassen)
    # 28 from 4x4x2 (4x Strassen)
    
    terms = []
    
    # Helper to generate 2x2x2 Strassen terms
    # Returns list of (u, v, w) where u,v,w are lists of 4 floats/ints
    # u corresponds to A (2x2), v to B (2x2), w to C (2x2)
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    def get_strassen_222():
        # M1 = (A00+A11)(B00+B11)
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # C00+=M1, C11+=M1
        
        # M2 = (A10+A11)B00
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, 0] # C10+=M2, C11-=M2
        
        # M3 = A00(B01-B11)
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        w3 = [0, 1, 0, 0] # C01+=M3, C11+=M3
        
        # M4 = A11(B10-B00)
        u4 = [0, 0, 0, 1]
        v4 = [0, 0, 1, -1]
        w4 = [1, 0, 1, 0] # C00+=M4, C10+=M4
        
        # M5 = (A00+A01)B11
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [0, 0, 0, 1] # C00-=M5, C01+=M5
        
        # M6 = (A10-A00)(B00+B01)
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1] # C11+=M6
        
        # M7 = (A01-A11)(B10+B11)
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, 0] # C00+=M7, C11+=M7
        
        return [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]
    
    base_222 = get_strassen_222()
    
    # Part 1: 4x4x4 decomposition (Rank 49)
    # This handles k in [0, 4)
    # We apply Strassen at the 4x4 level.
    # The 7 Strassen products P1..P7 are 2x2x2 multiplications.
    # We need to map the 2x2 blocks to the 4x4 indices.
    
    # Strassen steps for 4x4:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # For each Pk, we have a 2x2x2 multiplication.
    # We expand each into 7 rank-1 terms.
    
    # Mapping of 2x2 block (r, c) to 4x4 indices:
    # Block (0,0) -> i in [0,2), j in [0,2)
    # Block (0,1) -> i in [0,2), j in [2,4)
    # Block (1,0) -> i in [2,4), j in [0,2)
    # Block (1,1) -> i in [2,4), j in [2,4)
    
    def map_block_to_indices(block_r, block_c, dim=2):
        start_i = block_r * dim
        start_j = block_c * dim
        return list(range(start_i, start_i+dim)), list(range(start_j, start_j+dim))
    
    # Define the linear combinations for the 7 Strassen products
    # Each product is (A_comb, B_comb, C_comb)
    # A_comb is a linear combo of A blocks
    # B_comb is a linear combo of B blocks
    
    strassen_444_steps = [
        # (A_blocks_coeffs, B_blocks_coeffs, C_blocks_coeffs)
        # A_blocks: 0:A00, 1:A01, 2:A10, 3:A11
        # B_blocks: 0:B00, 1:B01, 2:B10, 3:B11
        # C_blocks: 0:C00, 1:C01, 2:C10, 3:C11
        
        # P1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # P3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # P4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # P5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        # P6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # P7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Generate 49 terms
    for a_coeffs, b_coeffs, c_coeffs in strassen_444_steps:
        # For each of the 7 base 222 terms
        for u_base, v_base, w_base in base_222:
            u_term = [0.0] * 16
            v_term = [0.0] * 16
            w_term = [0.0] * 16
            
            # Expand u_base (2x2) into 4x4 using a_coeffs
            # u_base corresponds to the "inner" 2x2 multiplication
            # a_coeffs selects which 2x2 blocks of A are involved
            # Actually, the Strassen step computes (A_comb) * (B_comb).
            # A_comb is a 2x2 matrix formed by summing A blocks.
            # So if A_comb = A00 + A11, then the "u" vector for the 2x2 multiplication
            # is applied to A00 and A11.
            
            # The 2x2 multiplication is between A_comb (2x2) and B_comb (2x2).
            # The result is C_comb (2x2).
            # The rank-1 decomposition of this 2x2 multiplication is given by base_222.
            # u_base is the coefficient for A_comb's entries.
            # v_base is the coefficient for B_comb's entries.
            # w_base is the coefficient for C_comb's entries.
            
            # So for each base term (u_base, v_base, w_base):
            # u_term gets contributions from A blocks where a_coeffs is non-zero.
            # The value is a_coeffs[block] * u_base[inner_idx].
            
            # Inner idx 0:00, 1:01, 2:10, 3:11 in the 2x2 matrix.
            # Map to 4x4 indices.
            
            for inner_idx in range(4):
                # Determine which A block this inner idx belongs to?
                # No, u_base is a vector of length 4 for a 2x2 matrix.
                # The 2x2 matrix IS the combination.
                # So u_base[0] multiplies the (0,0) entry of A_comb.
                # A_comb(0,0) = sum(a_coeffs[b] * A_b(0,0)).
                
                # This logic is slightly wrong.
                # Strassen P1 = (A00+A11)(B00+B11).
                # This is a 2x2 multiplication.
                # Let X = A00+A11, Y = B00+B11.
                # We decompose X*Y into 7 rank-1 terms.
                # Term 1: u1 * v1 * w1.
                # u1 is a vector for X. v1 for Y. w1 for Z=X*Y.
                # So u1[0] * X[0,0] etc.
                # X[0,0] = A00[0,0] + A11[0,0].
                # So u_term[global_idx_for_A00_00] += u1[0] * 1 (coeff for A00).
                # u_term[global_idx_for_A11_00] += u1[0] * 1 (coeff for A11).
                
                # So we iterate over the blocks involved in the combination.
                
                # For u_term:
                for b_idx in range(4):
                    coeff_a = a_coeffs[b_idx]
                    if coeff_a == 0: continue
                    # b_idx 0:A00, 1:A01, 2:A10, 3:A11
                    # Inner idx 0:00, 1:01, 2:10, 3:11
                    # Global idx for A block b_idx, inner entry inner_idx
                    # Block b_idx has row r = b_idx // 2, col c = b_idx % 2
                    r = b_idx // 2
                    c = b_idx % 2
                    # Inner entry inner_idx has row ri = inner_idx // 2, col ci = inner_idx % 2
                    ri = inner_idx // 2
                    ci = inner_idx % 2
                    # Global i = r*2 + ri, j = c*2 + ci
                    gi = r * 2 + ri
                    gj = c * 2 + ci
                    u_term[gi * 4 + gj] += coeff_a * u_base[inner_idx]
                
                # For v_term:
                for b_idx in range(4):
                    coeff_b = b_coeffs[b_idx]
                    if coeff_b == 0: continue
                    r = b_idx // 2
                    c = b_idx % 2
                    ri = inner_idx // 2
                    ci = inner_idx % 2
                    gi = r * 2 + ri
                    gj = c * 2 + ci
                    # B indices: j, k. In 4x4x4, B is 4x4.
                    # B block b_idx corresponds to j-block r, k-block c?
                    # In Strassen for 4x4, B is split into 2x2 blocks of 2x2 matrices.
                    # B00 (j in 0-1, k in 0-1), B01 (j in 0-1, k in 2-3), etc.
                    # So same mapping.
                    v_term[gi * 4 + gj] += coeff_b * v_base[inner_idx]
                
                # For w_term:
                for c_idx in range(4):
                    coeff_c = c_coeffs[c_idx]
                    if coeff_c == 0: continue
                    r = c_idx // 2
                    c = c_idx % 2
                    ri = inner_idx // 2
                    ci = inner_idx % 2
                    # C indices: k, i.
                    # C block c_idx corresponds to k-block r, i-block c?
                    # C00 (k in 0-1, i in 0-1), C01 (k in 0-1, i in 2-3), etc.
                    # So same mapping.
                    gi = r * 2 + ri
                    gj = c * 2 + ci
                    # W index: k * n + i. Here n=4.
                    # k = gi, i = gj.
                    w_term[gi * 4 + gj] += coeff_c * w_base[inner_idx]
            
            terms.append((u_term, v_term, w_term))
            
    # Part 2: 4x4x2 decomposition (Rank 28)
    # Handles k in [4, 6)
    # 4 independent 2x2x2 multiplications.
    
    # P1: A00 * B_top -> C_top_left (k in 4-5, i in 0-1)
    # P2: A01 * B_bottom -> C_top_left
    # P3: A10 * B_top -> C_bottom_left (k in 4-5, i in 2-3)
    # P4: A11 * B_bottom -> C_bottom_left
    
    # B_top is B[0:2, 4:6]. B_bottom is B[2:4, 4:6].
    
    def add_222_term(a_block, b_rows, c_block, u_base, v_base, w_base):
        # a_block: 0:A00, 1:A01, 2:A10, 3:A11
        # b_rows: 0:Top(0-1), 1:Bottom(2-3)
        # c_block: 0:TopLeft(i 0-1), 1:BottomLeft(i 2-3)
        
        u_term = [0.0] * 16
        v_term = [0.0] * 24 # 4x6
        w_term = [0.0] * 24 # 6x4
        
        r_a = a_block // 2
        c_a = a_block % 2
        
        r_b = b_rows * 2 # 0 or 2
        # k is 4,5. So k_offset = 4.
        
        r_c = c_block * 2 # 0 or 2 (for i)
        # k is 4,5.
        
        for inner_idx in range(4):
            ri = inner_idx // 2
            ci = inner_idx % 2
            
            # A
            gi_a = r_a * 2 + ri
            gj_a = c_a * 2 + ci
            u_term[gi_a * 4 + gj_a] = u_base[inner_idx]
            
            # B
            gi_b = r_b + ri
            gj_b = 4 + ci # k in 4,5
            v_term[gi_b * 6 + gj_b] = v_base[inner_idx]
            
            # C
            gi_c = 4 + ri # k in 4,5
            gj_c = r_c + ci # i in 0-1 or 2-3
            w_term[gi_c * 4 + gj_c] = w_base[inner_idx]
            
        terms.append((u_term, v_term, w_term))
        
    # Generate 28 terms
    for u_base, v_base, w_base in base_222:
        add_222_term(0, 0, 0, u_base, v_base, w_base) # A00 * B_top -> C_topleft
        add_222_term(1, 1, 0, u_base, v_base, w_base) # A01 * B_bottom -> C_topleft
        add_222_term(2, 0, 1, u_base, v_base, w_base) # A10 * B_top -> C_bottomleft
        add_222_term(3, 1, 1, u_base, v_base, w_base) # A11 * B_bottom -> C_bottomleft

    # Construct U, V, W
    rank = len(terms)
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(24)]
    
    for col, (u, v, w) in enumerate(terms):
        for i in range(16):
            U[i][col] = u[i]
        for i in range(24):
            V[i][col] = v[i]
            W[i][col] = w[i]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
