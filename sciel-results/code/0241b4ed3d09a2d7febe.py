import numpy as np

def construct():
    n = 5
    m = 5
    p = 5
    
    # We will build the U, V, W matrices.
    # U shape: (25, rank)
    # V shape: (25, rank)
    # W shape: (25, rank)
    
    terms = []
    
    # Helper to add a term
    # u_indices: list of (index, coeff)
    # v_indices: list of (index, coeff)
    # w_indices: list of (index, coeff)
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))

    # Indices for A (input 1): row * 5 + col
    # Indices for B (input 2): row * 5 + col
    # Indices for C (output): row * 5 + col
    
    # 1. Compute A_11 * B_11 using Strassen 4x4
    # A_11 is A[0:4, 0:4] -> indices 0..19 (excluding 4, 9, 14, 19? No, 0..15 is 4x4)
    # A_11 indices: r in 0..3, c in 0..3 -> r*5+c
    # B_11 indices: r in 0..3, c in 0..3 -> r*5+c
    # C_11 indices: r in 0..3, c in 0..3 -> r*5+c
    
    # Strassen 4x4 generates 49 terms for 4x4 matrices.
    # We need to map the local 4x4 indices to global 5x5 indices.
    # Local (r, c) in 4x4 -> Global (r, c) in 5x5.
    # Local index l = r*4+c. Global index g = r*5+c.
    
    # Let's hardcode the 49 terms for 4x4 Strassen or generate them.
    # Generating them recursively is safer.
    
    def strassen_2x2_terms():
        # Returns list of (u_dict, v_dict, w_dict) for 2x2 multiplication
        # Local indices 0..3
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B10-B11)
        # M4 = (A11)(B01-B10)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # A00=0, A01=1, A10=2, A11=3
        # B00=0, B01=1, B10=2, B11=3
        
        t1 = ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}) # C00 += M1 - M4 + M5 - M7? No.
        # C00 = M1 - M4 + M5 - M7
        # C01 = M2 + M4
        # C10 = M3 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the 7 multiplications
        mults = [
            ({0:1, 3:1}, {0:1, 3:1}), # M1
            ({2:1, 3:1}, {0:1}),      # M2
            ({0:1}, {2:1, 3:-1}),     # M3
            ({3:1}, {1:1, 2:-1}),     # M4
            ({0:1, 1:1}, {3:1}),      # M5
            ({2:1, 0:-1}, {0:1, 1:1}),# M6
            ({1:1, 3:-1}, {2:1, 3:1}) # M7
        ]
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M2 + M4
        # C10 = M3 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # We need to combine these into terms.
        # Each term in the final 2x2 decomposition is a linear combination of the 7 mults.
        # But the "rank" is the number of scalar multiplications.
        # So we have 7 terms.
        # Term 1 (M1): U={0:1, 3:1}, V={0:1, 3:1}, W={0:1, 3:1} (contributes to C00, C11)
        # Wait, W is the output matrix.
        # M1 contributes to C00 (coeff 1) and C11 (coeff 1).
        # So W dict for Term 1 is {0:1, 3:1}.
        
        term_1 = ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        
        # M2 contributes to C01 (coeff 1) and C11 (coeff -1).
        # W dict for Term 2 is {1:1, 3:-1}.
        term_2 = ({2:1, 3:1}, {0:1}, {1:1, 3:-1})
        
        # M3 contributes to C10 (coeff 1) and C11 (coeff 1).
        # W dict for Term 3 is {2:1, 3:1}.
        term_3 = ({0:1}, {2:1, 3:-1}, {2:1, 3:1})
        
        # M4 contributes to C00 (coeff -1) and C01 (coeff 1).
        # W dict for Term 4 is {0:-1, 1:1}.
        term_4 = ({3:1}, {1:1, 2:-1}, {0:-1, 1:1})
        
        # M5 contributes to C00 (coeff 1).
        # W dict for Term 5 is {0:1}.
        term_5 = ({0:1, 1:1}, {3:1}, {0:1})
        
        # M6 contributes to C11 (coeff 1).
        # W dict for Term 6 is {3:1}.
        term_6 = ({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        
        # M7 contributes to C00 (coeff -1) and C10 (coeff 1).
        # W dict for Term 7 is {0:-1, 2:1}.
        term_7 = ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 2:1})
        
        return [term_1, term_2, term_3, term_4, term_5, term_6, term_7]

    def strassen_4x4_terms():
        # Recursively apply strassen_2x2 to 4 blocks.
        # Blocks: 00, 01, 10, 11
        # Local 4x4 indices:
        # Block 00: 0,1,4,5
        # Block 01: 2,3,6,7
        # Block 10: 8,9,12,13
        # Block 11: 10,11,14,15
        
        # We need to map the 2x2 Strassen terms (which operate on blocks) to 4x4 entries.
        # A 2x2 Strassen term has U, V, W dicts with keys 0..3 (block indices).
        # We need to expand these into 4x4 indices.
        
        base_terms = strassen_2x2_terms()
        
        expanded_terms = []
        
        # Map block index (0..3) to 4x4 local indices
        # Block 0 (00): local 0,1,4,5
        # Block 1 (01): local 2,3,6,7
        # Block 2 (10): local 8,9,12,13
        # Block 3 (11): local 10,11,14,15
        
        def expand_block_dict(d, block_map):
            # d: dict {block_idx: coeff}
            # block_map: list of 4 lists of local indices
            # Returns dict {local_idx: coeff}
            new_d = {}
            for b_idx, coeff in d.items():
                for l_idx in block_map[b_idx]:
                    new_d[l_idx] = new_d.get(l_idx, 0) + coeff
            return new_d
        
        block_maps = [
            [0, 1, 4, 5], # Block 00
            [2, 3, 6, 7], # Block 01
            [8, 9, 12, 13], # Block 10
            [10, 11, 14, 15] # Block 11
        ]
        
        # This simple expansion is incorrect for Strassen.
        # Strassen on 4x4 involves computing 7 products of 2x2 blocks.
        # Each 2x2 block product is computed using 7 scalar multiplications.
        # So total 49 scalar multiplications.
        # The "terms" in the 4x4 decomposition are the 49 scalar multiplications.
        # We need to generate them.
        
        # Let's do it properly.
        # The 7 block-level terms from strassen_2x2_terms() define how the 4 blocks are combined.
        # For each of the 7 block terms, we perform a 2x2 matrix multiplication.
        # The input to this 2x2 multiplication is not scalar, but 2x2 matrices.
        # So we need to apply strassen_2x2 again to the "effective" 2x2 matrices?
        # No, the "effective" matrices are sums of blocks.
        # Example: M1 = (A00+A11)(B00+B11).
        # Let X = A00+A11, Y = B00+B11.
        # X and Y are 2x2 matrices.
        # We compute Z = X*Y.
        # Z is a 2x2 matrix.
        # We can compute Z using the 7-term Strassen for 2x2 matrices.
        # Each term in Z's Strassen decomposition is a scalar multiplication of linear combinations of entries of X and Y.
        # Entries of X are sums of entries of A00, A11.
        # So each scalar multiplication in Z corresponds to a linear combination of entries of A00, A11, B00, B11.
        
        # So, for each of the 7 outer terms (M1..M7):
        #   Identify the block coefficients for U and V.
        #   Construct the 2x2 "effective" U and V matrices where each entry is a linear combination of 4x4 entries.
        #   Apply strassen_2x2_terms to these effective matrices?
        #   No, strassen_2x2_terms returns dicts of local indices 0..3.
        #   We need to "lift" these local indices to the 4x4 global indices.
        
        # Let's define the lifting.
        # For a given outer term (e.g. M1), U_blocks = {0:1, 3:1}.
        # This means U_eff[0,0] = A00[0,0] + A11[0,0]? No.
        # A00 is a 2x2 matrix. A11 is a 2x2 matrix.
        # U_eff is a 2x2 matrix.
        # U_eff[r,c] = sum of A_block[r,c] * coeff.
        # So U_eff[0,0] = A00[0,0] + A11[0,0].
        # U_eff[0,1] = A00[0,1] + A11[0,1].
        # etc.
        
        # Now we apply strassen_2x2 to U_eff and V_eff.
        # The terms from strassen_2x2_terms() have U_dict, V_dict, W_dict with keys 0..3 (indices of U_eff, V_eff).
        # We need to expand these keys to 4x4 indices.
        # Key 0 in U_eff corresponds to (0,0) in the 2x2 block structure.
        # This corresponds to entries (0,0) and (0,1) of A00 and A11?
        # No, U_eff is a 2x2 matrix. Its entries are U_eff[0,0], U_eff[0,1], U_eff[1,0], U_eff[1,1].
        # These are linear combinations of the 4x4 entries.
        # Specifically, U_eff[0,0] = A00[0,0] + A11[0,0].
        # U_eff[0,1] = A00[0,1] + A11[0,1].
        # U_eff[1,0] = A00[1,0] + A11[1,0].
        # U_eff[1,1] = A00[1,1] + A11[1,1].
        
        # So when strassen_2x2_terms says "use U entry 0 with coeff 1", it means use U_eff[0,0] with coeff 1.
        # Which means use (A00[0,0] + A11[0,0]) with coeff 1.
        # So we add A00[0,0] and A11[0,0] to the U dict of the final term.
        
        # We need a mapping from "effective 2x2 index" (0..3) to "4x4 indices".
        # Eff 0 -> (0,0) of blocks.
        # Eff 1 -> (0,1) of blocks.
        # Eff 2 -> (1,0) of blocks.
        # Eff 3 -> (1,1) of blocks.
        
        # For a given outer term with U_blocks dict:
        # We build a mapping eff_idx -> list of (global_4x4_idx, coeff).
        
        for outer_term in base_terms:
            u_blocks, v_blocks, w_blocks = outer_term
            
            # Build mapping for U
            u_map = {0: [], 1: [], 2: [], 3: []}
            for b_idx, coeff in u_blocks.items():
                # b_idx is 0..3
                # b_idx 0 -> Block 00. Entries 0,1,4,5.
                # b_idx 1 -> Block 01. Entries 2,3,6,7.
                # b_idx 2 -> Block 10. Entries 8,9,12,13.
                # b_idx 3 -> Block 11. Entries 10,11,14,15.
                
                # We need to map eff_idx (0..3) to specific entries in these blocks.
                # Eff 0 corresponds to (0,0) in the 2x2 block grid.
                # So for Block 00, entry (0,0) is global 4x4 index 0.
                # For Block 01, entry (0,0) is global 4x4 index 2.
                # For Block 10, entry (0,0) is global 4x4 index 8.
                # For Block 11, entry (0,0) is global 4x4 index 10.
                
                # Let's create a helper to get the 4x4 index for a block and local (r,c)
                def get_4x4_idx(block, r, c):
                    br = block // 2
                    bc = block % 2
                    return (br * 2 + r) * 4 + (bc * 2 + c)
                
                # Eff 0 is (0,0). Eff 1 is (0,1). Eff 2 is (1,0). Eff 3 is (1,1).
                eff_to_local = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
                
                for eff_idx in range(4):
                    er, ec = eff_to_local[eff_idx]
                    idx4 = get_4x4_idx(b_idx, er, ec)
                    u_map[eff_idx].append((idx4, coeff))
            
            # Similarly for V
            v_map = {0: [], 1: [], 2: [], 3: []}
            for b_idx, coeff in v_blocks.items():
                for eff_idx in range(4):
                    er, ec = eff_to_local[eff_idx]
                    idx4 = get_4x4_idx(b_idx, er, ec)
                    v_map[eff_idx].append((idx4, coeff))
            
            # Now apply strassen_2x2 to the effective matrices
            for inner_term in base_terms:
                i_u, i_v, i_w = inner_term
                
                final_u = {}
                final_v = {}
                final_w = {}
                
                # Expand U
                for eff_idx, coeff in i_u.items():
                    for idx4, c in u_map[eff_idx]:
                        final_u[idx4] = final_u.get(idx4, 0) + coeff * c
                
                # Expand V
                for eff_idx, coeff in i_v.items():
                    for idx4, c in v_map[eff_idx]:
                        final_v[idx4] = final_v.get(idx4, 0) + coeff * c
                
                # Expand W
                # W from inner term is for the 2x2 result of the effective multiplication.
                # This result is added to the 4x4 result at the block positions defined by w_blocks.
                # Wait, the outer term w_blocks defines which blocks of the 4x4 result receive the 2x2 result.
                # For M1, w_blocks is {0:1, 3:1}.
                # So the 2x2 result Z is added to Block 00 and Block 11.
                # Z[0,0] goes to Block 00[0,0] (4x4 idx 0) and Block 11[0,0] (4x4 idx 10).
                # Z[0,1] goes to Block 00[0,1] (4x4 idx 1) and Block 11[0,1] (4x4 idx 11).
                # etc.
                
                for eff_idx, coeff in i_w.items():
                    er, ec = eff_to_local[eff_idx]
                    for b_idx, w_coeff in w_blocks.items():
                        idx4 = get_4x4_idx(b_idx, er, ec)
                        final_w[idx4] = final_w.get(idx4, 0) + coeff * w_coeff
                
                if final_u and final_v and final_w:
                    expanded_terms.append((final_u, final_v, final_w))
        
        return expanded_terms

    terms_4x4 = strassen_4x4_terms()
    
    # Map 4x4 local indices to 5x5 global indices
    # 4x4 index l = r*4+c. Global index g = r*5+c.
    def map_4x4_to_5x5(d):
        new_d = {}
        for l_idx, coeff in d.items():
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 5 + c
            new_d[g_idx] = new_d.get(g_idx, 0) + coeff
        return new_d

    for u, v, w in terms_4x4:
        gu = map_4x4_to_5x5(u)
        gv = map_4x4_to_5x5(v)
        gw = map_4x4_to_5x5(w)
        terms.append((gu, gv, gw))

    # Add terms for the 5th row/col interactions
    # C11 (4x4) already computed.
    # Need:
    # C12 (4x1): A11 * b12 + a12 * b22
    # C21 (1x4): a21 * B11 + a22 * b21
    # C22 (1x1): a21 * b12 + a22 * b22
    
    # Indices:
    # A11: 0..15 (4x4)
    # a12: A[0..3, 4] -> 4, 9, 14, 19
    # a21: A[4, 0..3] -> 20, 21, 22, 23
    # a22: A[4, 4] -> 24
    
    # B11: 0..15
    # b12: B[0..3, 4] -> 4, 9, 14, 19
    # b21: B[4, 0..3] -> 20, 21, 22, 23
    # b22: B[4, 4] -> 24
    
    # C12: C[0..3, 4] -> 4, 9, 14, 19
    # C21: C[4, 0..3] -> 20, 21, 22, 23
    # C22: C[4, 4] -> 24
    
    # Term for A11 * b12 (4 terms, one for each row of C12)
    # C12[r] = sum_c A11[r,c] * b12[c]
    for r in range(4):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        c_idx = r * 5 + 4
        w_dict[c_idx] = 1
        for c in range(4):
            a_idx = r * 5 + c
            b_idx = c * 5 + 4 # b12[c] is B[c, 4]
            u_dict[a_idx] = 1
            v_dict[b_idx] = 1
        terms.append((u_dict, v_dict, w_dict))
        
    # Term for a12 * b22 (1 term)
    # C12[r] += a12[r] * b22
    # This can be combined with the previous terms? No, different structure.
    # Actually, we can just add 4 separate terms or 1 term?
    # a12 * b22 is a vector scaled by a scalar.
    # C12[r] = a12[r] * b22.
    # This is 4 scalar multiplications? No, b22 is shared.
    # But in tensor decomposition, each term is a rank-1 tensor.
    # We can have one term for each r?
    # Or one term where U has a12 entries, V has b22, W has C12 entries?
    # U: {4:1, 9:1, 14:1, 19:1}
    # V: {24:1}
    # W: {4:1, 9:1, 14:1, 19:1}
    # This is 1 term.
    u_dict = {4:1, 9:1, 14:1, 19:1}
    v_dict = {24:1}
    w_dict = {4:1, 9:1, 14:1, 19:1}
    terms.append((u_dict, v_dict, w_dict))
    
    # Term for a21 * B11 (4 terms, one for each col of C21)
    # C21[c] = sum_r a21[r] * B11[r,c]
    for c in range(4):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        c_idx = 4 * 5 + c
        w_dict[c_idx] = 1
        for r in range(4):
            a_idx = 4 * 5 + r # a21[r] is A[4, r]
            b_idx = r * 5 + c # B11[r,c]
            u_dict[a_idx] = 1
            v_dict[b_idx] = 1
        terms.append((u_dict, v_dict, w_dict))
        
    # Term for a22 * b21 (1 term)
    # C21[c] += a22 * b21[c]
    u_dict = {24:1}
    v_dict = {20:1, 21:1, 22:1, 23:1}
    w_dict = {20:1, 21:1, 22:1, 23:1}
    terms.append((u_dict, v_dict, w_dict))
    
    # Term for a21 * b12 (1 term)
    # C22 = sum_r a21[r] * b12[r]
    u_dict = {20:1, 21:1, 22:1, 23:1}
    v_dict = {4:1, 9:1, 14:1, 19:1}
    w_dict = {24:1}
    terms.append((u_dict, v_dict, w_dict))
    
    # Term for a22 * b22 (1 term)
    u_dict = {24:1}
    v_dict = {24:1}
    w_dict = {24:1}
    terms.append((u_dict, v_dict, w_dict))
    
    # Build matrices
    rank = len(terms)
    U = np.zeros((25, rank))
    V = np.zeros((25, rank))
    W = np.zeros((25, rank))
    
    for k, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx, k] = coeff
        for idx, coeff in v_dict.items():
            V[idx, k] = coeff
        for idx, coeff in w_dict.items():
            W[idx, k] = coeff
            
    return {'U': U, 'V': V, 'W': W}
