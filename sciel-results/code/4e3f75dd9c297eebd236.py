import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    # We will build the rank-1 tensors and collect them
    # Each tensor is a tuple (u_vec, v_vec, w_vec)
    # u_vec is length 8
    # v_vec is length 28
    # w_vec is length 14
    
    tensors = []
    
    def add_term(u_indices, v_indices, w_indices, coeffs):
        """
        Adds a single rank-1 term.
        u_indices: list of (index, coeff) for U
        v_indices: list of (index, coeff) for V
        w_indices: list of (index, coeff) for W
        """
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        
        for idx, c in u_indices:
            u[idx] += c
        for idx, c in v_indices:
            v[idx] += c
        for idx, c in w_indices:
            w[idx] += c
            
        tensors.append((u, v, w))

    # Helper to get U index: i * m + j
    def u_idx(i, j):
        return i * m + j

    # Helper to get V index: j * p + k
    def v_idx(j, k):
        return j * p + k

    # Helper to get W index: k * n + i
    def w_idx(k, i):
        return k * n + i

    # --- Part 1: A (2x4) * B1 (4x4) using Strassen ---
    # A blocks (1x2 vectors):
    # A11: i=0, j=0,1
    # A12: i=0, j=2,3
    # A21: i=1, j=0,1
    # A22: i=1, j=2,3
    
    # B1 blocks (2x2 matrices):
    # B11: j=0,1; k=0,1
    # B12: j=0,1; k=2,3
    # B21: j=2,3; k=0,1
    # B22: j=2,3; k=2,3
    
    # We define the 7 Strassen multiplications.
    # Each multiplication is of a 1x2 vector (U_part) and a 2x2 matrix (V_part).
    # The result is a 1x2 vector R.
    # R[0] and R[1] are linear combinations of the 4 scalar products.
    # We need to map these scalar products to the global W indices based on the Strassen combination.
    
    # Strassen Formulas for 2x2 block multiplication:
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5 - M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5
    
    # Here C is 2x4.
    # C[0, 0:2] corresponds to C11 (block row 0, col 0) -> W indices for k=0,1; i=0
    # C[0, 2:4] corresponds to C12 (block row 0, col 1) -> W indices for k=2,3; i=0
    # C[1, 0:2] corresponds to C21 (block row 1, col 0) -> W indices for k=0,1; i=1
    # C[1, 2:4] corresponds to C22 (block row 1, col 1) -> W indices for k=2,3; i=1
    
    # Let's define the U and V parts for each M_k.
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # For each M_k, we have a U_vec (1x2) and V_mat (2x2).
    # The product R = U_vec * V_mat is 1x2.
    # R[0] = U_vec[0]*V_mat[0,0] + U_vec[1]*V_mat[1,0]
    # R[1] = U_vec[0]*V_mat[0,1] + U_vec[1]*V_mat[1,1]
    
    # We will generate 4 terms for each M_k.
    # Term 1: U_vec[0] * V_mat[0,0] -> contributes to R[0]
    # Term 2: U_vec[1] * V_mat[1,0] -> contributes to R[0]
    # Term 3: U_vec[0] * V_mat[0,1] -> contributes to R[1]
    # Term 4: U_vec[1] * V_mat[1,1] -> contributes to R[1]
    
    # We need to define the coefficients for U_vec and V_mat for each M_k.
    
    # M1
    u_m1 = [(u_idx(0,0), 1), (u_idx(0,1), 1), (u_idx(1,2), 1), (u_idx(1,3), 1)] # A11 + A22
    v_m1 = [(v_idx(0,0), 1), (v_idx(0,1), 1), (v_idx(1,0), 1), (v_idx(1,1), 1), 
            (v_idx(2,2), 1), (v_idx(2,3), 1), (v_idx(3,2), 1), (v_idx(3,3), 1)] # B11 + B22
    
    # M2
    u_m2 = [(u_idx(1,0), 1), (u_idx(1,1), 1), (u_idx(1,2), 1), (u_idx(1,3), 1)] # A21 + A22
    v_m2 = [(v_idx(0,0), 1), (v_idx(0,1), 1), (v_idx(1,0), 1), (v_idx(1,1), 1)] # B11
    
    # M3
    u_m3 = [(u_idx(0,0), 1), (u_idx(0,1), 1)] # A11
    v_m3 = [(v_idx(0,2), 1), (v_idx(0,3), 1), (v_idx(1,2), 1), (v_idx(1,3), 1), 
            (v_idx(2,2), -1), (v_idx(2,3), -1), (v_idx(3,2), -1), (v_idx(3,3), -1)] # B12 - B22
    
    # M4
    u_m4 = [(u_idx(1,2), 1), (u_idx(1,3), 1)] # A22
    v_m4 = [(v_idx(2,0), 1), (v_idx(2,1), 1), (v_idx(3,0), 1), (v_idx(3,1), 1), 
            (v_idx(0,0), -1), (v_idx(0,1), -1), (v_idx(1,0), -1), (v_idx(1,1), -1)] # B21 - B11
    
    # M5
    u_m5 = [(u_idx(0,0), 1), (u_idx(0,1), 1), (u_idx(0,2), 1), (u_idx(0,3), 1)] # A11 + A12
    v_m5 = [(v_idx(2,2), 1), (v_idx(2,3), 1), (v_idx(3,2), 1), (v_idx(3,3), 1)] # B22
    
    # M6
    u_m6 = [(u_idx(1,0), 1), (u_idx(1,1), 1), (u_idx(0,0), -1), (u_idx(0,1), -1)] # A21 - A11
    v_m6 = [(v_idx(0,0), 1), (v_idx(0,1), 1), (v_idx(1,0), 1), (v_idx(1,1), 1), 
            (v_idx(0,2), 1), (v_idx(0,3), 1), (v_idx(1,2), 1), (v_idx(1,3), 1)] # B11 + B12
    
    # M7
    u_m7 = [(u_idx(0,2), 1), (u_idx(0,3), 1), (u_idx(1,2), -1), (u_idx(1,3), -1)] # A12 - A22
    v_m7 = [(v_idx(2,0), 1), (v_idx(2,1), 1), (v_idx(3,0), 1), (v_idx(3,1), 1), 
            (v_idx(2,2), 1), (v_idx(2,3), 1), (v_idx(3,2), 1), (v_idx(3,3), 1)] # B21 + B22
    
    # We need to map the results R_k[0] and R_k[1] to the final W indices.
    # R_k is 1x2.
    # R_k[0] corresponds to the first column of the block result.
    # R_k[1] corresponds to the second column of the block result.
    
    # For M1:
    # R1[0] contributes to C11[0] (k=0, i=0) and C21[0] (k=0, i=1)?
    # No, Strassen computes C11, C12, C21, C22 which are 2x2 blocks.
    # C11 is 2x2. C11[0,0] is C[0,0]. C11[0,1] is C[0,1].
    # Wait, my A blocks are 1x2. So the "2x2" block multiplication result is actually 2x4.
    # The Strassen algorithm I'm using is for 2x2 blocks where the blocks themselves are matrices.
    # Here, the "blocks" of A are 1x2 vectors. The "blocks" of B are 2x2 matrices.
    # The result blocks are 1x2 vectors.
    # So C11 is a 1x2 vector? No.
    # Let's re-verify the dimensions.
    # A is 2x4. Split into 2x2 blocks of size 1x2.
    # A = [[A11, A12], [A21, A22]] where A11 is 1x2.
    # B is 4x4. Split into 2x2 blocks of size 2x2.
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2.
    # Product C = A B is 2x4.
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11 B11 + A12 B21. Dimensions: (1x2)(2x2) + (1x2)(2x2) = 1x2.
    # So C11 is a 1x2 vector.
    # C12 is a 1x2 vector.
    # C21 is a 1x2 vector.
    # C22 is a 1x2 vector.
    
    # So R_k is a 1x2 vector.
    # R_k[0] is the first element of the vector (corresponds to column 0 of the block).
    # R_k[1] is the second element (corresponds to column 1 of the block).
    
    # Mapping R_k to C:
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5 - M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5
    
    # C11[0] (k=0, i=0) = M1[0] - M4[0] + M6[0]
    # C11[1] (k=1, i=0) = M1[1] - M4[1] + M6[1]
    # C12[0] (k=2, i=0) = M3[0] + M5[0] - M7[0]
    # C12[1] (k=3, i=0) = M3[1] + M5[1] - M7[1]
    # C21[0] (k=0, i=1) = M2[0] + M4[0]
    # C21[1] (k=1, i=1) = M2[1] + M4[1]
    # C22[0] (k=2, i=1) = M1[0] - M3[0] + M5[0]
    # C22[1] (k=3, i=1) = M1[1] - M3[1] + M5[1]
    
    # We will iterate through the 7 M's.
    # For each M, we have 4 terms.
    # Term 1: U0 * V00 -> R[0]
    # Term 2: U1 * V10 -> R[0]
    # Term 3: U0 * V01 -> R[1]
    # Term 4: U1 * V11 -> R[1]
    
    # We need the coefficients of R[0] and R[1] in the final C.
    
    def add_strassen_term(m_idx, u_coeffs, v_coeffs, r0_coeffs, r1_coeffs):
        # u_coeffs: list of (u_idx, coeff)
        # v_coeffs: list of (v_idx, coeff)
        # r0_coeffs: dict mapping (k, i) to coeff for R[0]
        # r1_coeffs: dict mapping (k, i) to coeff for R[1]
        
        # Term 1: U0 * V00
        u0_idx, u0_c = u_coeffs[0]
        v00_idx, v00_c = v_coeffs[0]
        w_indices = []
        for (k, i), c in r0_coeffs.items():
            w_indices.append((w_idx(k, i), c))
        add_term([(u0_idx, u0_c)], [(v00_idx, v00_c)], w_indices)
        
        # Term 2: U1 * V10
        u1_idx, u1_c = u_coeffs[1]
        v10_idx, v10_c = v_coeffs[1]
        w_indices = []
        for (k, i), c in r0_coeffs.items():
            w_indices.append((w_idx(k, i), c))
        add_term([(u1_idx, u1_c)], [(v10_idx, v10_c)], w_indices)
        
        # Term 3: U0 * V01
        u0_idx, u0_c = u_coeffs[0]
        v01_idx, v01_c = v_coeffs[2]
        w_indices = []
        for (k, i), c in r1_coeffs.items():
            w_indices.append((w_idx(k, i), c))
        add_term([(u0_idx, u0_c)], [(v01_idx, v01_c)], w_indices)
        
        # Term 4: U1 * V11
        u1_idx, u1_c = u_coeffs[1]
        v11_idx, v11_c = v_coeffs[3]
        w_indices = []
        for (k, i), c in r1_coeffs.items():
            w_indices.append((w_idx(k, i), c))
        add_term([(u1_idx, u1_c)], [(v11_idx, v11_c)], w_indices)

    # Define the R0 and R1 contributions for each M
    # M1: R0 in C11[0] (+1), C22[0] (+1). R1 in C11[1] (+1), C22[1] (+1).
    r0_m1 = {(0,0): 1, (2,1): 1}
    r1_m1 = {(1,0): 1, (3,1): 1}
    add_strassen_term(1, u_m1, v_m1, r0_m1, r1_m1)
    
    # M2: R0 in C21[0] (+1). R1 in C21[1] (+1).
    r0_m2 = {(0,1): 1}
    r1_m2 = {(1,1): 1}
    add_strassen_term(2, u_m2, v_m2, r0_m2, r1_m2)
    
    # M3: R0 in C12[0] (+1), C22[0] (-1). R1 in C12[1] (+1), C22[1] (-1).
    r0_m3 = {(2,0): 1, (2,1): -1}
    r1_m3 = {(3,0): 1, (3,1): -1}
    add_strassen_term(3, u_m3, v_m3, r0_m3, r1_m3)
    
    # M4: R0 in C11[0] (-1), C21[0] (+1). R1 in C11[1] (-1), C21[1] (+1).
    r0_m4 = {(0,0): -1, (0,1): 1}
    r1_m4 = {(1,0): -1, (1,1): 1}
    add_strassen_term(4, u_m4, v_m4, r0_m4, r1_m4)
    
    # M5: R0 in C12[0] (+1), C22[0] (+1). R1 in C12[1] (+1), C22[1] (+1).
    r0_m5 = {(2,0): 1, (2,1): 1}
    r1_m5 = {(3,0): 1, (3,1): 1}
    add_strassen_term(5, u_m5, v_m5, r0_m5, r1_m5)
    
    # M6: R0 in C11[0] (+1). R1 in C11[1] (+1).
    r0_m6 = {(0,0): 1}
    r1_m6 = {(1,0): 1}
    add_strassen_term(6, u_m6, v_m6, r0_m6, r1_m6)
    
    # M7: R0 in C12[0] (-1). R1 in C12[1] (-1).
    r0_m7 = {(2,0): -1}
    r1_m7 = {(3,0): -1}
    add_strassen_term(7, u_m7, v_m7, r0_m7, r1_m7)

    # --- Part 2: A (2x4) * B2 (4x3) using Split ---
    # A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B2_L = B2[0:2, :], B2_R = B2[2:4, :]
    # P_L = A_L B2_L, P_R = A_R B2_R
    # C2 = P_L + P_R
    
    # We need to compute 2x2 by 2x3 multiplication.
    # Split B2_L (2x3) into B2_L_2 (2x2, cols 0-1) and B2_L_1 (2x1, col 2).
    # P_L_2 = A_L B2_L_2 (2x2 by 2x2, Rank 7)
    # P_L_1 = A_L B2_L_1 (2x2 by 2x1, Rank 4)
    
    # Similarly for P_R.
    
    # Helper for 2x2 by 2x2 Strassen
    # A (2x2), B (2x2).
    # A indices in global U:
    # A_L: i=0,1; j=0,1.
    # A_R: i=0,1; j=2,3.
    
    # B indices in global V:
    # B2_L_2: j=0,1; k=0,1.
    # B2_L_1: j=0,1; k=2.
    # B2_R_2: j=2,3; k=0,1.
    # B2_R_1: j=2,3; k=2.
    
    # W indices for C2 (2x3):
    # C2[0,0] (k=0, i=0)
    # C2[0,1] (k=1, i=0)
    # C2[0,2] (k=2, i=0)
    # C2[1,0] (k=0, i=1)
    # C2[1,1] (k=1, i=1)
    # C2[1,2] (k=2, i=1)
    
    # We will write a generic function to add a 2x2 by 2x2 Strassen decomposition.
    # Parameters:
    # u_base: list of 4 u_indices for A (row 0 col 0, row 0 col 1, row 1 col 0, row 1 col 1)
    # v_base: list of 4 v_indices for B (row 0 col 0, row 0 col 1, row 1 col 0, row 1 col 1)
    # w_map: dict mapping (k, i) to coeff for C11, C12, C21, C22?
    # Actually, the result of 2x2 by 2x2 is a 2x2 matrix.
    # We need to map its 4 entries to the global W indices.
    
    def add_2x2_strassen(u_indices, v_indices, w_coeffs):
        """
        u_indices: [u00, u01, u10, u11]
        v_indices: [v00, v01, v10, v11]
        w_coeffs: dict {(k,i): coeff} for each of the 4 output positions?
        No, Strassen computes C11, C12, C21, C22.
        C11 is at (k0, i0), C12 at (k1, i0), C21 at (k0, i1), C22 at (k1, i1).
        We need to pass the mapping for these 4 positions.
        """
        u00, u01, u10, u11 = u_indices
        v00, v01, v10, v11 = v_indices
        
        # M1 = (A00+A11)(B00+B11)
        u_m1 = [(u00, 1), (u11, 1)]
        v_m1 = [(v00, 1), (v11, 1)]
        # M2 = (A10+A11)B00
        u_m2 = [(u10, 1), (u11, 1)]
        v_m2 = [(v00, 1)]
        # M3 = A00(B01-B11)
        u_m3 = [(u00, 1)]
        v_m3 = [(v01, 1), (v11, -1)]
        # M4 = A11(B10-B00)
        u_m4 = [(u11, 1)]
        v_m4 = [(v10, 1), (v00, -1)]
        # M5 = (A00+A01)B11
        u_m5 = [(u00, 1), (u01, 1)]
        v_m5 = [(v11, 1)]
        # M6 = (A10-A00)(B00+B01)
        u_m6 = [(u10, 1), (u00, -1)]
        v_m6 = [(v00, 1), (v01, 1)]
        # M7 = (A01-A11)(B10+B11)
        u_m7 = [(u01, 1), (u11, -1)]
        v_m7 = [(v10, 1), (v11, 1)]
        
        # For each M, we have 4 terms (since A and B are 2x2, the "block" multiplication is scalar).
        # Wait, for 2x2 by 2x2, the "blocks" are scalars.
        # So each M is a scalar product.
        # The result C is 2x2.
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5 - M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5
        
        # We need to add 7 terms, each corresponding to one M.
        # Each term is u_vec * v_vec * w_vec.
        # u_vec has 2 non-zeros (from u_mX).
        # v_vec has 1 or 2 non-zeros (from v_mX).
        # w_vec has non-zeros at the positions corresponding to C00, C01, C10, C11.
        
        # Let's refine add_term for this case.
        # We'll just call add_term 7 times.
        
        # M1
        w_m1 = []
        for (k, i), c in w_coeffs.items():
             # We need to know which C entry this M contributes to.
             # This is getting complicated. Let's hardcode the 7 terms for 2x2x2.
             pass

    # Given the complexity, it's safer to just expand the 2x2x2 Strassen terms manually in the code.
    # But I already have 28 terms from Part 1.
    # I need 22 more terms.
    
    # Let's implement a helper to add a scalar product term.
    # This is basically what add_term does.
    
    # Let's define the 7 terms for a 2x2x2 Strassen block.
    # We will call this function twice (for P_L and P_R).
    
    def add_2x2x2_strassen_block(u_idxs, v_idxs, w_map):
        # u_idxs: [u00, u01, u10, u11]
        # v_idxs: [v00, v01, v10, v11]
        # w_map: {(0,0): coeff, (0,1): coeff, (1,0): coeff, (1,1): coeff}
        # Note: (k,i) here refers to the local 2x2 output.
        
        u00, u01, u10, u11 = u_idxs
        v00, v01, v10, v11 = v_idxs
        
        # M1
        u1 = [(u00, 1), (u11, 1)]
        v1 = [(v00, 1), (v11, 1)]
        w1 = [(w_map[(0,0)], 1), (w_map[(1,1)], 1)] # C00 and C11
        add_term(u1, v1, w1)
        
        # M2
        u2 = [(u10, 1), (u11, 1)]
        v2 = [(v00, 1)]
        w2 = [(w_map[(1,0)], 1)] # C10
        add_term(u2, v2, w2)
        
        # M3
        u3 = [(u00, 1)]
        v3 = [(v01, 1), (v11, -1)]
        w3 = [(w_map[(0,1)], 1), (w_map[(1,1)], -1)] # C01 and C11
        add_term(u3, v3, w3)
        
        # M4
        u4 = [(u11, 1)]
        v4 = [(v10, 1), (v00, -1)]
        w4 = [(w_map[(0,0)], -1), (w_map[(1,0)], 1)] # C00 and C10
        add_term(u4, v4, w4)
        
        # M5
        u5 = [(u00, 1), (u01, 1)]
        v5 = [(v11, 1)]
        w5 = [(w_map[(0,1)], 1), (w_map[(1,1)], 1)] # C01 and C11
        add_term(u5, v5, w5)
        
        # M6
        u6 = [(u10, 1), (u00, -1)]
        v6 = [(v00, 1), (v01, 1)]
        w6 = [(w_map[(0,0)], 1)] # C00
        add_term(u6, v6, w6)
        
        # M7
        u7 = [(u01, 1), (u11, -1)]
        v7 = [(v10, 1), (v11, 1)]
        w7 = [(w_map[(0,1)], -1)] # C01
        add_term(u7, v7, w7)

    # P_L = A_L B2_L
    # A_L u_idxs: [u(0,0), u(0,1), u(1,0), u(1,1)]
    # B2_L_2 v_idxs: [v(0,0), v(0,1), v(1,0), v(1,1)]
    # Output C2_L is 2x2.
    # C2_L[0,0] -> C2[0,0] (k=0, i=0)
    # C2_L[0,1] -> C2[0,1] (k=1, i=0)
    # C2_L[1,0] -> C2[1,0] (k=0, i=1)
    # C2_L[1,1] -> C2[1,1] (k=1, i=1)
    
    u_L = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_L2 = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    w_map_L = {(0,0): w_idx(0,0), (0,1): w_idx(1,0), (1,0): w_idx(0,1), (1,1): w_idx(1,1)}
    add_2x2x2_strassen_block(u_L, v_L2, w_map_L)
    
    # P_L_1 = A_L B2_L_1 (2x2 by 2x1)
    # Naive: 4 terms.
    # C2_L_1 is 2x1.
    # C2_L_1[0] -> C2[0,2] (k=2, i=0)
    # C2_L_1[1] -> C2[1,2] (k=2, i=1)
    
    # Term 1: A[0,0] * B[0,2] -> C[0,2]
    add_term([(u_idx(0,0), 1)], [(v_idx(0,2), 1)], [(w_idx(2,0), 1)])
    # Term 2: A[0,1] * B[1,2] -> C[0,2]
    add_term([(u_idx(0,1), 1)], [(v_idx(1,2), 1)], [(w_idx(2,0), 1)])
    # Term 3: A[1,0] * B[0,2] -> C[1,2]
    add_term([(u_idx(1,0), 1)], [(v_idx(0,2), 1)], [(w_idx(2,1), 1)])
    # Term 4: A[1,1] * B[1,2] -> C[1,2]
    add_term([(u_idx(1,1), 1)], [(v_idx(1,2), 1)], [(w_idx(2,1), 1)])
    
    # P_R = A_R B2_R
    # A_R u_idxs: [u(0,2), u(0,3), u(1,2), u(1,3)]
    # B2_R_2 v_idxs: [v(2,0), v(2,1), v(3,0), v(3,1)]
    
    u_R = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_R2 = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    w_map_R = {(0,0): w_idx(0,0), (0,1): w_idx(1,0), (1,0): w_idx(0,1), (1,1): w_idx(1,1)}
    add_2x2x2_strassen_block(u_R, v_R2, w_map_R)
    
    # P_R_1 = A_R B2_R_1 (2x2 by 2x1)
    # B2_R_1 is col 2 of B2_R, which is col 2 of B2, which is col 2 of B.
    # B2 is cols 4,5,6 of B. So B2_L_1 is col 4 of B?
    # Wait. B is 4x7.
    # B1 is 4x4 (cols 0-3).
    # B2 is 4x3 (cols 4-6).
    # B2_L is rows 0-1 of B2.
    # B2_L_1 is col 2 of B2_L, which is col 6 of B?
    # No, B2_L is 2x3. Cols 0,1,2 of B2.
    # B2_L_2 is cols 0,1 of B2 (cols 4,5 of B).
    # B2_L_1 is col 2 of B2 (col 6 of B).
    
    # So v_idx for B2_L_1 is v(0,6) and v(1,6).
    # And for B2_R_1 is v(2,6) and v(3,6).
    
    # Term 1: A[0,2] * B[0,6] -> C[0,2]
    add_term([(u_idx(0,2), 1)], [(v_idx(0,6), 1)], [(w_idx(2,0), 1)])
    # Term 2: A[0,3] * B[1,6] -> C[0,2]
    add_term([(u_idx(0,3), 1)], [(v_idx(1,6), 1)], [(w_idx(2,0), 1)])
    # Term 3: A[1,2] * B[0,6] -> C[1,2]
    add_term([(u_idx(1,2), 1)], [(v_idx(0,6), 1)], [(w_idx(2,1), 1)])
    # Term 4: A[1,3] * B[1,6] -> C[1,2]
    add_term([(u_idx(1,3), 1)], [(v_idx(1,6), 1)], [(w_idx(2,1), 1)])
    
    # Total terms: 28 + 7 + 4 + 7 + 4 = 50.
    
    rank = len(tensors)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(tensors):
        for i in range(n * m):
            U[i][r] = u[i]
        for i in range(m * p):
            V[i][r] = v[i]
        for i in range(p * n):
            W[i][r] = w[i]
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
