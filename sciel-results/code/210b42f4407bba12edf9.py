import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in two parts:
    # Part 1: M_{4,4,4} using Strassen (Rank 49)
    # Part 2: M_{4,4,3} using Block Method (Rank 44)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # --- Part 1: M_{4,4,4} ---
    # Strassen decomposition for 4x4x4
    # This consists of 7 sub-decompositions of M_{2,2,2}
    # Each M_{2,2,2} has rank 7.
    # We need to map the indices correctly.
    
    # Helper to get Strassen 2x2x2 U, V, W
    def get_strassen_222():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices for U (A): 0,1,2,3 correspond to A00, A01, A10, A11
        # Indices for V (B): 0,1,2,3 correspond to B00, B01, B10, B11
        # Indices for W (C): 0,1,2,3 correspond to C00, C01, C10, C11
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[1,0] = 1
        V[0,0] = 1; V[1,0] = 1
        W[0,0] = 1; W[3,0] = 1 # C00 += M1, C11 += M1
        
        # M2 = (A10+A11)B00
        U[1,1] = 1; U[2,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = 1 # C10 += M2, C11 -= M2
        
        # M3 = A00(B01-B11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[0,2] = 1; W[2,2] = 1 # C00 += M3, C01 += M3
        
        # M4 = A11(B10-B00)
        U[1,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1 # C00 += M4, C10 += M4
        
        # M5 = (A00+A01)B11
        U[0,4] = 1; U[2,4] = 1
        V[3,4] = 1
        W[0,4] = 1; W[3,4] = 1 # C00 -= M5, C11 += M5
        
        # M6 = (A10-A00)(B00+B01)
        U[1,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[2,5] = 1
        W[2,5] = 1; W[3,5] = 1 # C01 += M6, C11 += M6
        
        # M7 = (A01-A11)(B10+B11)
        U[2,6] = 1; U[1,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[2,6] = 1 # C00 += M7, C10 += M7
        
        return U, V, W

    # We need to embed 2x2x2 into 4x4x4.
    # A is 4x4, B is 4x4.
    # A blocks: A00 (0,1;0,1), A01 (0,1;2,3), A10 (2,3;0,1), A11 (2,3;2,3)
    # B blocks: B00 (0,1;0,1), B01 (0,1;2,3), B10 (2,3;0,1), B11 (2,3;2,3)
    # C blocks: C00 (0,1;0,1), C01 (0,1;2,3), C10 (2,3;0,1), C11 (2,3;2,3)
    
    # Mapping from 2x2 indices to 4x4 indices
    # A: 0->0, 1->1, 2->4, 3->5 (for A00)
    # Actually, let's just construct the full 16x7, 16x7, 16x7 matrices for the 4x4x4 part.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # 7 Strassen products
    # Each product k (0..6) uses 7 terms.
    # Total 49 terms.
    
    # Define block mappings
    # A indices:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    # B indices:
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    # C indices:
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # We need to map the local 2x2 indices (0,1,2,3) to global 4x4 indices.
    # Local 0 -> Global 0
    # Local 1 -> Global 1
    # Local 2 -> Global 4
    # Local 3 -> Global 5
    # This is for the "00" block.
    # For "01" block:
    # Local 0 -> Global 2
    # Local 1 -> Global 3
    # Local 2 -> Global 6
    # Local 3 -> Global 7
    
    # Let's define a function to expand a 2x2x2 decomposition into 4x4x4 for a specific block combination.
    
    def expand_222_to_444(U_222, V_222, W_222, a_block, b_block, c_block):
        # a_block, b_block, c_block are 0,1,2,3 corresponding to 00,01,10,11
        # Returns expanded U, V, W of size 16x7
        U_exp = np.zeros((16, 7))
        V_exp = np.zeros((16, 7))
        W_exp = np.zeros((16, 7))
        
        # Map local indices to global indices
        # Local row/col in 2x2: 0,1
        # Global row/col in 4x4:
        # Block 00: rows 0,1; cols 0,1
        # Block 01: rows 0,1; cols 2,3
        # Block 10: rows 2,3; cols 0,1
        # Block 11: rows 2,3; cols 2,3
        
        def get_global_indices(block):
            if block == 0: # 00
                return [0, 1, 4, 5]
            elif block == 1: # 01
                return [2, 3, 6, 7]
            elif block == 2: # 10
                return [8, 9, 12, 13]
            else: # 11
                return [10, 11, 14, 15]
        
        a_idx = get_global_indices(a_block)
        b_idx = get_global_indices(b_block)
        c_idx = get_global_indices(c_block)
        
        for col in range(7):
            for local_i in range(4):
                if U_222[local_i, col] != 0:
                    U_exp[a_idx[local_i], col] = U_222[local_i, col]
                if V_222[local_i, col] != 0:
                    V_exp[b_idx[local_i], col] = V_222[local_i, col]
                if W_222[local_i, col] != 0:
                    W_exp[c_idx[local_i], col] = W_222[local_i, col]
                    
        return U_exp, V_exp, W_exp

    U_222, V_222, W_222 = get_strassen_222()
    
    # Strassen combinations for 4x4x4
    # M1: A00+A11, B00+B11 -> C00, C11
    # This is a sum of two 2x2x2 products? No.
    # Strassen for 4x4 uses 7 products of 2x2 matrices.
    # Let's call the 2x2 blocks A00, A01, A10, A11.
    # M1 = (A00+A11)(B00+B11)
    # This is not a single 2x2x2 tensor. It is a sum of 4 1x1x1 products?
    # No, we treat A00+A11 as a 2x2 matrix.
    # So M1 is a 2x2x2 multiplication.
    # The input A for this 2x2x2 is A00+A11.
    # The input B for this 2x2x2 is B00+B11.
    # The output C contributes to C00 and C11.
    
    # We need to construct the U, V, W for the term "M1" in the 4x4x4 decomposition.
    # U for M1:
    # Rows corresponding to A00 get +1.
    # Rows corresponding to A11 get +1.
    # V for M1:
    # Rows corresponding to B00 get +1.
    # Rows corresponding to B11 get +1.
    # W for M1:
    # Rows corresponding to C00 get +1.
    # Rows corresponding to C11 get +1.
    
    # We can do this by summing the expanded 2x2x2 decompositions.
    # But wait, the 2x2x2 decomposition of M1 uses the *sum* of the matrices.
    # So we can just add the U, V, W of the 2x2x2 decomposition for A00, B00, C00
    # and the U, V, W of the 2x2x2 decomposition for A11, B11, C11.
    # Since the decomposition is linear, we can sum the matrices.
    
    # Let's generate the 7 terms for 4x4x4.
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # We add:
    # - Expansion of 222 for A00, B00, C00
    # - Expansion of 222 for A11, B11, C11
    # But we must be careful. The 222 decomposition computes (A00+A11)(B00+B11).
    # If we just sum the decompositions for A00B00 and A11B11, we get A00B00 + A11B11.
    # We need A00B00 + A00B11 + A11B00 + A11B11.
    # Strassen computes this with 7 products.
    # The 7 products are linear combinations.
    # So we need to construct the U, V, W for each of the 7 Strassen products of the 4x4x4.
    # Each of these 7 products is a 2x2x2 multiplication.
    # For example, P1 = (A00+A11)(B00+B11).
    # U for P1:
    #   A00 rows: +1
    #   A11 rows: +1
    # V for P1:
    #   B00 rows: +1
    #   B11 rows: +1
    # W for P1:
    #   C00 rows: +1
    #   C11 rows: +1
    
    # We can create a helper to build U, V, W for a single 2x2x2 product term
    # given the linear combination coefficients for A, B, C.
    
    def build_222_term(a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: dict mapping block (0..3) to coeff
        # b_coeffs: dict mapping block (0..3) to coeff
        # c_coeffs: dict mapping block (0..3) to coeff
        U_t = np.zeros((16, 7))
        V_t = np.zeros((16, 7))
        W_t = np.zeros((16, 7))
        
        # We need to add the 222 decomposition scaled by the coefficients?
        # No, the 222 decomposition is for the product of the *sums*.
        # If A_sum = sum(ca_i * A_i), B_sum = sum(cb_j * B_j).
        # Then Product = A_sum B_sum.
        # The U matrix for this product has rows for A_i scaled by ca_i.
        # The V matrix has rows for B_j scaled by cb_j.
        # The W matrix has rows for C_k scaled by ck_k.
        
        # We use the standard 222 U, V, W (from get_strassen_222) as a template.
        # The template assumes A, B, C are 2x2 matrices.
        # Here, the "2x2 matrix" A_sum is formed by blocks.
        
        # So for each local index l in 0..3:
        # U_exp[global_idx(l), col] += sum( ca_i * U_222[l, col] ) for i where block i contains global_idx(l)?
        # No, global_idx(l) belongs to a specific block.
        
        # Let's iterate over the 7 columns of the 222 decomposition.
        for col in range(7):
            # U part
            for block in range(4):
                if block in a_coeffs:
                    coeff = a_coeffs[block]
                    if coeff != 0:
                        a_idx = get_global_indices(block)
                        for l in range(4):
                            U_t[a_idx[l], col] += coeff * U_222[l, col]
            
            # V part
            for block in range(4):
                if block in b_coeffs:
                    coeff = b_coeffs[block]
                    if coeff != 0:
                        b_idx = get_global_indices(block)
                        for l in range(4):
                            V_t[b_idx[l], col] += coeff * V_222[l, col]
                        
            # W part
            for block in range(4):
                if block in c_coeffs:
                    coeff = c_coeffs[block]
                    if coeff != 0:
                        c_idx = get_global_indices(block)
                        for l in range(4):
                            W_t[c_idx[l], col] += coeff * W_222[l, col]
                            
        return U_t, V_t, W_t

    # Define the 7 Strassen terms for 4x4x4
    # P1: (A0+A3)(B0+B3) -> C0, C3
    # P2: (A2+A3)B0 -> C1, C3
    # P3: A0(B1-B3) -> C0, C1
    # P4: A3(B2-B0) -> C0, C1
    # P5: (A0+A1)B3 -> C0, C3
    # P6: (A2-A0)(B0+B1) -> C1, C3
    # P7: (A1-A3)(B2+B3) -> C0, C1
    
    # Note: My previous Strassen definition used different indices.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map these to blocks 0,1,2,3 (00,01,10,11).
    
    terms = []
    # M1
    terms.append({
        'a': {0: 1, 3: 1},
        'b': {0: 1, 3: 1},
        'c': {0: 1, 3: 1}
    })
    # M2
    terms.append({
        'a': {2: 1, 3: 1},
        'b': {0: 1},
        'c': {1: 1, 3: -1} # C10 += M2, C11 -= M2
    })
    # M3
    terms.append({
        'a': {0: 1},
        'b': {1: 1, 3: -1},
        'c': {0: 1, 1: 1} # C00 += M3, C01 += M3
    })
    # M4
    terms.append({
        'a': {3: 1},
        'b': {2: 1, 0: -1},
        'c': {0: 1, 1: 1} # C00 += M4, C10 += M4
    })
    # M5
    terms.append({
        'a': {0: 1, 1: 1},
        'b': {3: 1},
        'c': {0: -1, 3: 1} # C00 -= M5, C11 += M5
    })
    # M6
    terms.append({
        'a': {2: 1, 0: -1},
        'b': {0: 1, 1: 1},
        'c': {1: 1, 3: 1} # C01 += M6, C11 += M6
    })
    # M7
    terms.append({
        'a': {1: 1, 3: -1},
        'b': {2: 1, 3: 1},
        'c': {0: 1, 1: 1} # C00 += M7, C10 += M7
    })
    
    for t in terms:
        U_t, V_t, W_t = build_222_term(t['a'], t['b'], t['c'])
        U_444 = np.hstack([U_444, U_t])
        V_444 = np.hstack([V_444, V_t])
        W_444 = np.hstack([W_444, W_t])
        
    # Part 1 done. Rank 49.
    
    # --- Part 2: M_{4,4,3} ---
    # Rank 44.
    # 4 blocks of M_{2,2,3}.
    # Each M_{2,2,3} is M_{2,2,2} (Strassen) + M_{2,2,1} (Schoolbook).
    
    # We need to handle the 3 columns of B.
    # B is 4x3.
    # Split B into B1 (2x3) and B2 (2x3).
    # A is 4x4. Split into A00, A01, A10, A11 (2x2).
    # C1 = A00 B1 + A01 B2
    # C2 = A10 B1 + A11 B2
    
    # We compute 4 products:
    # P11: A00 B1
    # P12: A01 B2
    # P21: A10 B1
    # P22: A11 B2
    
    # Each P is 2x2x3.
    # Decompose P into P_strassen (2x2x2) + P_school (2x2x1).
    # P_strassen uses columns 0,1 of B.
    # P_school uses column 2 of B.
    
    # We need to construct U, V, W for M_{4,4,3}.
    # U: 16x44
    # V: 12x44 (4x3 matrix flattened? No, V is m x p x rank. m=4, p=3. So 12x44)
    # W: 12x44 (p x n x rank. p=3, n=4. So 12x44)
    
    # Wait, the interface requires U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # For M_{4,4,3}:
    # n=4, m=4, p=3.
    # U: 16 x rank
    # V: 12 x rank
    # W: 12 x rank
    
    # We will build 4 sets of (U, V, W) for the 4 blocks.
    # Each block has rank 11.
    
    U_443 = np.zeros((16, 44))
    V_443 = np.zeros((12, 44))
    W_443 = np.zeros((12, 44))
    
    # Helper for M_{2,2,3}
    def get_223_decomp():
        # Returns U (4x11), V (6x11), W (6x11)
        # U: 4x11
        # V: 6x11 (2x3 flattened)
        # W: 6x11 (3x2 flattened? No, W is p x n. p=2, n=2. So 4x11? No.)
        # For M_{2,2,3}:
        # n=2, m=2, p=3.
        # U: 4x11
        # V: 6x11
        # W: 6x11 (p*n = 3*2 = 6)
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Part 1: Strassen 2x2x2 on cols 0,1 of B.
        # B cols 0,1 form a 2x2 matrix.
        # U_222, V_222, W_222 from get_strassen_222
        U_s, V_s, W_s = get_strassen_222()
        
        # Copy to U, V, W
        # U: same
        U[:, 0:7] = U_s
        # V: cols 0,1 of B are indices 0,1,2,3 in 2x3 flattening?
        # V index: i*m + j. m=2.
        # B is 2x3.
        # B[0,0]=0, B[0,1]=1, B[0,2]=2
        # B[1,0]=3, B[1,1]=4, B[1,2]=5
        # Strassen uses B[0,0], B[0,1], B[1,0], B[1,1].
        # Indices 0,1,3,4.
        # V_s has rows 0,1,2,3 corresponding to B00, B01, B10, B11.
        # Map V_s rows to V rows.
        # V_s row 0 (B00) -> V row 0
        # V_s row 1 (B01) -> V row 1
        # V_s row 2 (B10) -> V row 3
        # V_s row 3 (B11) -> V row 4
        
        v_map = [0, 1, 3, 4]
        for i in range(4):
            V[v_map[i], 0:7] = V_s[i, 0:7]
            
        # W: C is 2x2.
        # W_s has rows 0,1,2,3 corresponding to C00, C01, C10, C11.
        # W index: k*n + i. n=2.
        # C is 2x2.
        # C[0,0]=0, C[0,1]=1
        # C[1,0]=2, C[1,1]=3
        # W_s row 0 (C00) -> W row 0
        # W_s row 1 (C01) -> W row 1
        # W_s row 2 (C10) -> W row 2
        # W_s row 3 (C11) -> W row 3
        W[:, 0:7] = W_s
        
        # Part 2: Schoolbook 2x2x1 on col 2 of B.
        # B col 2 is indices 2, 5.
        # A is 2x2.
        # C is 2x1 (part of 2x2 C? No, C is 2x3).
        # Wait, M_{2,2,3} output C is 2x3.
        # W is 6x11.
        # W rows:
        # C[0,0]=0, C[0,1]=1, C[0,2]=2
        # C[1,0]=3, C[1,1]=4, C[1,2]=5
        
        # Schoolbook for 2x2x1:
        # 4 terms.
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
        
        # Term 8: A00 B02
        U[0, 7] = 1
        V[2, 7] = 1
        W[2, 7] = 1
        
        # Term 9: A01 B12
        U[1, 8] = 1
        V[5, 8] = 1
        W[2, 8] = 1
        
        # Term 10: A10 B02
        U[2, 9] = 1
        V[2, 9] = 1
        W[5, 9] = 1
        
        # Term 11: A11 B12
        U[3, 10] = 1
        V[5, 10] = 1
        W[5, 10] = 1
        
        return U, V, W

    U_223, V_223, W_223 = get_223_decomp()
    
    # Now expand to 4x4x3
    # Blocks:
    # P11: A00 (block 0) * B1 (rows 0,1 of B)
    # P12: A01 (block 1) * B2 (rows 2,3 of B)
    # P21: A10 (block 2) * B1 (rows 0,1 of B)
    # P22: A11 (block 3) * B2 (rows 2,3 of B)
    
    # B is 4x3.
    # B1 is rows 0,1. Indices 0,1,2,3,4,5 in V (12x44).
    # B2 is rows 2,3. Indices 6,7,8,9,10,11 in V.
    
    # C is 4x3.
    # C1 is rows 0,1. Indices 0..5 in W.
    # C2 is rows 2,3. Indices 6..11 in W.
    
    # A is 4x4.
    # A00 rows 0,1,4,5
    # A01 rows 2,3,6,7
    # A10 rows 8,9,12,13
    # A11 rows 10,11,14,15
    
    # We need to map U_223, V_223, W_223 to the global indices.
    
    def expand_223_to_443(U_223, V_223, W_223, a_block, b_rows, c_rows):
        U_exp = np.zeros((16, 11))
        V_exp = np.zeros((12, 11))
        W_exp = np.zeros((12, 11))
        
        a_idx = get_global_indices(a_block)
        
        # V mapping
        # b_rows: 0 or 1 (for B1 or B2)
        # B1 rows 0,1 -> V indices 0..5
        # B2 rows 2,3 -> V indices 6..11
        # Local V indices 0..5 map to global V indices.
        # If b_rows == 0:
        #   Local 0 -> 0
        #   Local 1 -> 1
        #   Local 2 -> 2
        #   Local 3 -> 3
        #   Local 4 -> 4
        #   Local 5 -> 5
        # If b_rows == 1:
        #   Local 0 -> 6
        #   Local 1 -> 7
        #   Local 2 -> 8
        #   Local 3 -> 9
        #   Local 4 -> 10
        #   Local 5 -> 11
        
        v_offset = b_rows * 6
        
        # W mapping
        # c_rows: 0 or 1 (for C1 or C2)
        # C1 rows 0,1 -> W indices 0..5
        # C2 rows 2,3 -> W indices 6..11
        w_offset = c_rows * 6
        
        for col in range(11):
            for l in range(4):
                if U_223[l, col] != 0:
                    U_exp[a_idx[l], col] = U_223[l, col]
            for l in range(6):
                if V_223[l, col] != 0:
                    V_exp[v_offset + l, col] = V_223[l, col]
                if W_223[l, col] != 0:
                    W_exp[w_offset + l, col] = W_223[l, col]
                    
        return U_exp, V_exp, W_exp

    # P11: A00, B1, C1
    U_t, V_t, W_t = expand_223_to_443(U_223, V_223, W_223, 0, 0, 0)
    U_443 = np.hstack([U_443, U_t])
    V_443 = np.hstack([V_443, V_t])
    W_443 = np.hstack([W_443, W_t])
    
    # P12: A01, B2, C1
    U_t, V_t, W_t = expand_223_to_443(U_223, V_223, W_223, 1, 1, 0)
    U_443 = np.hstack([U_443, U_t])
    V_443 = np.hstack([V_443, V_t])
    W_443 = np.hstack([W_443, W_t])
    
    # P21: A10, B1, C2
    U_t, V_t, W_t = expand_223_to_443(U_223, V_223, W_223, 2, 0, 1)
    U_443 = np.hstack([U_443, U_t])
    V_443 = np.hstack([V_443, V_t])
    W_443 = np.hstack([W_443, W_t])
    
    # P22: A11, B2, C2
    U_t, V_t, W_t = expand_223_to_443(U_223, V_223, W_223, 3, 1, 1)
    U_443 = np.hstack([U_443, U_t])
    V_443 = np.hstack([V_443, V_t])
    W_443 = np.hstack([W_443, W_t])
    
    # Combine Part 1 and Part 2
    # Part 1: M_{4,4,4} (cols 0-3 of B, cols 0-3 of C)
    # Part 2: M_{4,4,3} (cols 4-6 of B, cols 4-6 of C)
    
    # We need to align V and W indices.
    # V for Part 1: 16x49. B is 4x4. Indices 0..15.
    # V for Part 2: 12x44. B is 4x3. Indices 0..11.
    # Total B is 4x7. Indices 0..27.
    # Part 1 uses B cols 0-3.
    # Part 2 uses B cols 4-6.
    
    # In V_444, rows 0..15 correspond to B[0,0]..B[3,3].
    # In V_443, rows 0..11 correspond to B[0,4]..B[3,6].
    
    # We need to shift V_443 rows by 16?
    # B index: i*m + j.
    # Part 1: j=0..3.
    # Part 2: j=4..6.
    # V_443 row 0 corresponds to B[0,4].
    # Global V index for B[0,4] is 0*4 + 4 = 4.
    # But V_443 row 0 is local index 0.
    # So we need to map V_443 rows to global V rows.
    
    # Global V size: 4*7 = 28.
    # V_444 has 16 rows.
    # V_443 has 12 rows.
    
    V_full = np.zeros((28, 93))
    V_full[0:16, 0:49] = V_444
    # Map V_443 rows to global rows
    # V_443 row l corresponds to B[i, j] where i = l // 3, j = l % 3.
    # Global index = i * 4 + (j + 4).
    for l in range(12):
        i = l // 3
        j = l % 3
        global_idx = i * 4 + (j + 4)
        V_full[global_idx, 49:93] = V_443[l, 0:44]
        
    # W for Part 1: 16x49. C is 4x4. Indices 0..15.
    # W for Part 2: 12x44. C is 4x3. Indices 0..11.
    # Global W size: 7*4 = 28.
    # W index: k*n + i.
    # Part 1: k=0..3.
    # Part 2: k=4..6.
    
    W_full = np.zeros((28, 93))
    W_full[0:16, 0:49] = W_444
    # Map W_443 rows to global rows
    # W_443 row l corresponds to C[i, j] where i = l // 3, j = l % 3.
    # Global index = (j + 4) * 4 + i.
    for l in range(12):
        i = l // 3
        j = l % 3
        global_idx = (j + 4) * 4 + i
        W_full[global_idx, 49:93] = W_443[l, 0:44]
        
    U_full = np.hstack([U_444, U_443])
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

