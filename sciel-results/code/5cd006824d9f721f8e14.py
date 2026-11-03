import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4 by 4x4 (cols 0-3 of B)
    # Rank 48 for 4x4 by 4x3 (cols 4-6 of B)
    rank_44 = 49
    rank_43 = 48
    total_rank = rank_44 + rank_43
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4 by 4x4 using Strassen (Rank 49) ---
    # B cols 0..3
    # A is 4x4. B1 is 4x4.
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to expand this to 4x4.
    # Each 2x2 multiplication is done via Strassen (Rank 7).
    # So we have 7 groups of 7 multiplications.
    
    col_offset = 0
    for s in range(7): # 7 Strassen steps
        # Determine the linear combinations of A and B blocks
        # A blocks: A11, A12, A21, A22 (each 2x2)
        # B blocks: B11, B12, B21, B22 (each 2x2)
        
        # We define the coefficients for A and B for this Strassen step
        # a_coeffs: dict of block -> coeff
        # b_coeffs: dict of block -> coeff
        
        if s == 0: # M1
            a_coeffs = {0: 1, 3: 1} # A11, A22
            b_coeffs = {0: 1, 3: 1} # B11, B22
        elif s == 1: # M2
            a_coeffs = {2: 1, 3: 1} # A21, A22
            b_coeffs = {0: 1}       # B11
        elif s == 2: # M3
            a_coeffs = {0: 1}       # A11
            b_coeffs = {1: 1, 3: -1} # B12, B22
        elif s == 3: # M4
            a_coeffs = {3: 1}       # A22
            b_coeffs = {2: 1, 0: -1} # B21, B11
        elif s == 4: # M5
            a_coeffs = {0: 1, 1: 1} # A11, A12
            b_coeffs = {3: 1}       # B22
        elif s == 5: # M6
            a_coeffs = {2: 1, 0: -1} # A21, A11
            b_coeffs = {0: 1, 1: 1} # B11, B12
        elif s == 6: # M7
            a_coeffs = {1: 1, 3: -1} # A12, A22
            b_coeffs = {2: 1, 3: 1} # B21, B22
        
        # Now, perform Strassen on the 2x2 blocks defined by these coeffs
        # The 2x2 Strassen steps:
        # S1: (A11+A22)(B11+B22)
        # S2: (A21+A22)B11
        # S3: A11(B12-B22)
        # S4: A22(B21-B11)
        # S5: (A11+A12)B22
        # S6: (A21-A11)(B11+B12)
        # S7: (A12-A22)(B21+B22)
        
        # We need to combine the outer Strassen coeffs with the inner Strassen coeffs.
        # This is getting complicated to hardcode.
        # Alternative: Just use the known Rank 49 coefficients if available?
        # Or, simpler: Use the schoolbook for 4x4 (Rank 64) and schoolbook for 4x3 (Rank 48)?
        # Total 112. No improvement.
        
        # Let's try a different split.
        # What if we use Rank 49 for 4x4 and Rank 48 for 4x3?
        # I need the explicit matrices.
        
        pass # Placeholder for complex logic
    
    # Given the complexity of hardcoding Rank 49, let's try a simpler improvement.
    # Is there a simpler fast algorithm?
    
    # Let's try Rank 49 for 4x4 by using the fact that we can compute it.
    # Actually, I can generate the Rank 49 decomposition programmatically using the Strassen logic.
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2 matrix multiplication (Rank 7)
        # A, B, C are 2x2.
        # U: 4x7, V: 4x7, W: 4x7
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        W_s[1, 0] = -1; W_s[2, 0] = -1
        
        # M2 = (A21+A22)B11
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1; W_s[3, 1] = 1
        
        # M3 = A11(B12-B22)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[0, 2] = 1; W_s[1, 2] = 1
        
        # M4 = A22(B21-B11)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[2, 3] = 1; W_s[3, 3] = 1
        
        # M5 = (A11+A12)B22
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = 1; W_s[2, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[1, 5] = 1; W_s[2, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1; W_s[3, 6] = 1
        
        return U_s, V_s, W_s

    # Now, construct 4x4 Rank 49.
    # A is 4x4. B is 4x4.
    # We use 7 Strassen multiplications of 2x2 blocks.
    # Each 2x2 multiplication has rank 7.
    # Total rank 49.
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # Map 4x4 indices to 2x2 block indices
    # Block 0: rows 0-1, cols 0-1 (A11)
    # Block 1: rows 0-1, cols 2-3 (A12)
    # Block 2: rows 2-3, cols 0-1 (A21)
    # Block 3: rows 2-3, cols 2-3 (A22)
    
    def block_idx(r, c):
        return (r // 2) * 2 + (c // 2)
    
    # Outer Strassen Coeffs
    outer_coeffs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    U_s, V_s, W_s = get_strassen_2x2()
    
    for s_outer in range(7):
        a_coeff, b_coeff = outer_coeffs[s_outer]
        for s_inner in range(7):
            col = s_outer * 7 + s_inner
            # Combine coefficients
            # For each 2x2 element in U_s, V_s, W_s
            # U_s[i, s_inner] is coeff for A element i in inner Strassen
            # We need to map A element i to 4x4 A elements.
            
            # Inner Strassen A elements: 0,1,2,3 correspond to 2x2 blocks.
            # But U_s is defined for 2x2 matrix elements.
            # U_s[0] is A11, U_s[1] is A12, U_s[2] is A21, U_s[3] is A22.
            # So for each inner step s_inner, we have coeffs for A11, A12, A21, A22.
            
            a_inner_coeffs = U_s[:, s_inner]
            b_inner_coeffs = V_s[:, s_inner]
            w_inner_coeffs = W_s[:, s_inner]
            
            # a_inner_coeffs[0] is coeff for A11 (block 0)
            # a_inner_coeffs[1] is coeff for A12 (block 1)
            # a_inner_coeffs[2] is coeff for A21 (block 2)
            # a_inner_coeffs[3] is coeff for A22 (block 3)
            
            for block_a in range(4):
                coeff_a = a_inner_coeffs[block_a]
                if coeff_a != 0:
                    # Multiply by outer coeff for this block
                    outer_a = a_coeff.get(block_a, 0)
                    if outer_a == 0: continue
                    total_coeff = coeff_a * outer_a
                    # Add to U_44
                    # Block block_a corresponds to 4 elements in A (4x4)
                    # Block 0: (0,0), (0,1), (1,0), (1,1)
                    # Block 1: (0,2), (0,3), (1,2), (1,3)
                    # Block 2: (2,0), (2,1), (3,0), (3,1)
                    # Block 3: (2,2), (2,3), (3,2), (3,3)
                    
                    r_start = (block_a // 2) * 2
                    c_start = (block_a % 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            r = r_start + dr
                            c = c_start + dc
                            idx = r * 4 + c
                            U_44[idx, col] += total_coeff
            
            for block_b in range(4):
                coeff_b = b_inner_coeffs[block_b]
                if coeff_b != 0:
                    outer_b = b_coeff.get(block_b, 0)
                    if outer_b == 0: continue
                    total_coeff = coeff_b * outer_b
                    # Add to V_44
                    r_start = (block_b // 2) * 2
                    c_start = (block_b % 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            r = r_start + dr
                            c = c_start + dc
                            idx = r * 4 + c
                            V_44[idx, col] += total_coeff
            
            for block_c in range(4):
                coeff_c = w_inner_coeffs[block_c]
                if coeff_c != 0:
                    # W is 4x4. Block c corresponds to 4 elements in C.
                    # C = M1 ... M7 combinations.
                    # The inner Strassen W_s gives the C block contributions.
                    # But we need to map the inner C blocks to the 4x4 C.
                    # The outer Strassen combines the 7 M_k results.
                    # M_k is a 2x2 matrix.
                    # C11 = M1 - M4 + M5 - M6
                    # C12 = M3 + M7
                    # C21 = M2 + M4
                    # C22 = M1 - M3 + M6 + M7
                    
                    # So for each outer step s_outer, we add the M_k contribution to the appropriate C blocks.
                    # The inner W_s gives the 2x2 result of M_k.
                    # We need to distribute this 2x2 result to the 4x4 C.
                    
                    # M_k is 2x2.
                    # Its contribution to C11, C12, C21, C22 depends on s_outer.
                    
                    # Let's pre-define the mapping from M_k to C blocks.
                    # c_map[s_outer][block_c] = coeff
                    # block_c is 0,1,2,3 for C11, C12, C21, C22.
                    
                    pass # Need to implement this mapping

    # This is getting too complex to debug in one shot.
    # Let's fall back to a simpler improvement.
    
    # What if I just use Rank 64 for 4x4 (Schoolbook) and Rank 48 for 4x3 (Schoolbook)?
    # Total 112. No improvement.
    
    # Is there a simple Rank < 112?
    # Yes, Rank 97 is possible.
    # But implementing Rank 49 is hard.
    
    # Let's try Rank 49 for 4x4 using a known library or simple implementation.
    # Actually, I can just return the schoolbook decomposition for now to ensure correctness,
    # and then try to improve.
    
    # Wait, the task asks for a "stronger algorithm".
    # If I can't easily implement Rank 49, I should try Rank 97 with a simpler 4x4 fast alg?
    # No, 49 is the standard fast alg.
    
    # Let's try to implement the Rank 49 correctly.
    
    # Mapping of M_k to C blocks:
    # C11 = M1 - M4 + M5 - M6
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    c_map = {
        0: {0: 1, 3: -1, 1: 0, 2: 0}, # M1 -> C11(1), C22(-1)
        1: {2: 1, 3: 0, 0: 0, 1: 0},   # M2 -> C21(1)
        2: {1: 1, 3: -1, 0: 0, 2: 0},  # M3 -> C12(1), C22(-1)
        3: {0: -1, 2: 1, 1: 0, 3: 0},  # M4 -> C11(-1), C21(1)
        4: {0: 1, 2: 1, 1: 0, 3: 0},   # M5 -> C11(1), C21(1)
        5: {0: -1, 3: 1, 1: 0, 2: 0},  # M6 -> C11(-1), C22(1)
        6: {1: 1, 3: 1, 0: 0, 2: 0}    # M7 -> C12(1), C22(1)
    }
    # Note: c_map keys are s_outer (0..6). Values are dict block_c -> coeff.
    # block_c: 0=C11, 1=C12, 2=C21, 3=C22.
    
    # Re-run the construction with this mapping.
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for s_outer in range(7):
        a_coeff, b_coeff = outer_coeffs[s_outer]
        for s_inner in range(7):
            col = s_outer * 7 + s_inner
            a_inner_coeffs = U_s[:, s_inner]
            b_inner_coeffs = V_s[:, s_inner]
            w_inner_coeffs = W_s[:, s_inner]
            
            for block_a in range(4):
                coeff_a = a_inner_coeffs[block_a]
                if coeff_a != 0:
                    outer_a = a_coeff.get(block_a, 0)
                    if outer_a == 0: continue
                    total_coeff = coeff_a * outer_a
                    r_start = (block_a // 2) * 2
                    c_start = (block_a % 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            r = r_start + dr
                            c = c_start + dc
                            idx = r * 4 + c
                            U_44[idx, col] += total_coeff
            
            for block_b in range(4):
                coeff_b = b_inner_coeffs[block_b]
                if coeff_b != 0:
                    outer_b = b_coeff.get(block_b, 0)
                    if outer_b == 0: continue
                    total_coeff = coeff_b * outer_b
                    r_start = (block_b // 2) * 2
                    c_start = (block_b % 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            r = r_start + dr
                            c = c_start + dc
                            idx = r * 4 + c
                            V_44[idx, col] += total_coeff
            
            for block_c in range(4):
                coeff_c = w_inner_coeffs[block_c]
                if coeff_c != 0:
                    # Map to C blocks
                    # c_map[s_outer] gives coeff for each C block
                    # But w_inner_coeffs is for the 2x2 result of M_k.
                    # The 2x2 result has 4 elements: C11, C12, C21, C22 of M_k.
                    # These map to the 4x4 C blocks.
                    # M_k's C11 maps to C11 of 4x4?
                    # Yes, M_k is a 2x2 block.
                    # So M_k[0,0] adds to C[0,0], C[0,1], C[1,0], C[1,1]?
                    # No, M_k is a 2x2 matrix.
                    # M_k[0,0] is the top-left element of M_k.
                    # This element contributes to C11 (top-left 2x2 block of C).
                    # Specifically, C11[0,0] += M_k[0,0].
                    
                    # So for each element in the 2x2 M_k result, we add to the corresponding 4x4 C element.
                    # The 2x2 M_k result elements are indexed by w_inner_coeffs.
                    # w_inner_coeffs[0] is M_k[0,0]
                    # w_inner_coeffs[1] is M_k[0,1]
                    # w_inner_coeffs[2] is M_k[1,0]
                    # w_inner_coeffs[3] is M_k[1,1]
                    
                    # These map to C blocks:
                    # M_k[0,0] -> C11 (block 0)
                    # M_k[0,1] -> C12 (block 1)
                    # M_k[1,0] -> C21 (block 2)
                    # M_k[1,1] -> C22 (block 3)
                    
                    c_block_idx = block_c # 0,1,2,3 corresponds to M_k[0,0], M_k[0,1], M_k[1,0], M_k[1,1]
                    
                    # Get the outer coeff for this C block
                    outer_c = c_map[s_outer].get(c_block_idx, 0)
                    if outer_c == 0: continue
                    
                    total_coeff = coeff_c * outer_c
                    
                    # Add to W_44
                    r_start = (c_block_idx // 2) * 2
                    c_start = (c_block_idx % 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            r = r_start + dr
                            c = c_start + dc
                            idx = r * 4 + c
                            W_44[idx, col] += total_coeff

    # Part 2: 4x4 by 4x3 (Schoolbook, Rank 48)
    # B cols 4..6
    # C cols 4..6
    # A is 4x4.
    
    U_43 = np.zeros((16, 48))
    V_43 = np.zeros((12, 48)) # m*p = 4*3 = 12
    W_43 = np.zeros((12, 48)) # p*n = 3*4 = 12
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_43[i*4+j, col] = 1
                V_43[j*3+k, col] = 1
                W_43[k*4+i, col] = 1
                col += 1
    
    # Combine
    U = np.hstack([U_44, U_43])
    V = np.hstack([V_44, V_43])
    W = np.hstack([W_44, W_43])
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
