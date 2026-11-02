import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x4 * 4x6
    # Split i into 2 blocks of 1
    # Split j into 2 blocks of 2
    # Split k into 2 blocks of 3
    
    # Sub-tensor: 1x2 * 2x3
    # Rank 6
    # U: 1x2 -> (2, 6)
    # V: 2x3 -> (6, 6)
    # W: 3x1 -> (3, 6)
    
    # We need to build the full U, V, W
    # U shape: (2*4, R) = (8, R)
    # V shape: (4*6, R) = (24, R)
    # W shape: (6*2, R) = (12, R)
    
    # Strassen factors:
    # M1: (A11 + A00) * (B00 + B11)
    # M2: (A10 + A11) * B00
    # M3: A00 * (B01 - B11)
    # M4: A11 * (B10 - B00)
    # M5: (A00 + A11) * (B11 + B10)
    # M6: (A10 - A00) * (B00 + B01)
    # M7: (A11 - A10) * (B11 + B10)
    
    # Indices:
    # i: 0, 1
    # j: 0,1 (J0), 2,3 (J1)
    # k: 0,1,2 (K0), 3,4,5 (K1)
    
    # A blocks:
    # A00: i=0, j=0,1
    # A01: i=0, j=2,3
    # A10: i=1, j=0,1
    # A11: i=1, j=2,3
    
    # B blocks:
    # B00: j=0,1; k=0,1,2
    # B01: j=0,1; k=3,4,5
    # B10: j=2,3; k=0,1,2
    # B11: j=2,3; k=3,4,5
    
    # C blocks:
    # C00: i=0, k=0,1,2
    # C01: i=0, k=3,4,5
    # C10: i=1, k=0,1,2
    # C11: i=1, k=3,4,5
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 1x2 * 2x3 product.
    # We can define a helper to get the U, V, W for a specific 1x2 * 2x3 multiplication.
    # But the U, V, W matrices will have specific rows/cols corresponding to the blocks.
    
    # Let's create the base rank-6 decomposition for 1x2 * 2x3.
    # U_base: (2, 6)
    # V_base: (6, 6)
    # W_base: (3, 6)
    
    # For 1x2 * 2x3:
    # A is 1x2, B is 2x3.
    # C is 1x3.
    # C_k = A_0 B_0k + A_1 B_1k.
    # Rank 6 decomposition:
    # r=0..5.
    # U_base[j, r] = 1 if r//3 == j
    # V_base[j*3+k, r] = 1 if r%3 == k and r//3 == j
    # W_base[k, r] = 1 if r%3 == k
    
    U_base = np.zeros((2, 6))
    V_base = np.zeros((6, 6))
    W_base = np.zeros((3, 6))
    for r in range(6):
        j = r // 3
        k = r % 3
        U_base[j, r] = 1
        V_base[j*3+k, r] = 1
        W_base[k, r] = 1
        
    # Now construct full U, V, W
    # R = 7 * 6 = 42
    R = 42
    U = np.zeros((8, R))
    V = np.zeros((24, R))
    W = np.zeros((12, R))
    
    # Helper to add a block
    # add_block(M_idx, A_sign, A_i, A_j_start, B_sign, B_j_start, B_k_start, C_sign, C_i, C_k_start)
    # This is getting complex to map.
    
    # Let's just iterate over the 7 Strassen products and the 6 base ranks.
    
    # M1: (A11 + A00) * (B00 + B11)
    # A11: i=1, j=2,3. A00: i=0, j=0,1.
    # B00: j=0,1, k=0,1,2. B11: j=2,3, k=3,4,5.
    
    # We need to map the local indices of the 1x2 * 2x3 tensor to the global indices.
    
    # Local j: 0,1. Local k: 0,1,2.
    
    def add_product(m_idx, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: list of (i, j_start, sign)
        # b_coeffs: list of (j_start, k_start, sign)
        # c_coeffs: list of (i, k_start, sign)
        
        for r_local in range(6):
            r = m_idx * 6 + r_local
            
            # U part
            for i, j_start, sign in a_coeffs:
                for j_local in range(2):
                    j_global = j_start + j_local
                    U[i*4 + j_global, r] += sign * U_base[j_local, r_local]
                    
            # V part
            for j_start, k_start, sign in b_coeffs:
                for j_local in range(2):
                    for k_local in range(3):
                        j_global = j_start + j_local
                        k_global = k_start + k_local
                        V[j_global*6 + k_global, r] += sign * V_base[j_local*3 + k_local, r_local]
                        
            # W part
            for i, k_start, sign in c_coeffs:
                for k_local in range(3):
                    k_global = k_start + k_local
                    W[k_global*2 + i, r] += sign * W_base[k_local, r_local]

    # M1: (A11 + A00) * (B00 + B11)
    # A: A11 (i=1, j=2,3) + A00 (i=0, j=0,1)
    # B: B00 (j=0,1, k=0,1,2) + B11 (j=2,3, k=3,4,5)
    # C: C00 (i=0, k=0,1,2) + C10 (i=1, k=0,1,2) ?
    # Wait, C00 = M1 + ... and C10 = M2 + M4.
    # M1 contributes to C00 and C10?
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C01 = M3 + M5
    
    # So M1 contributes to C00 (i=0, k=0..2) and C11 (i=1, k=3..5)?
    # No, Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 is (A11+A00)(B00+B11).
    # This product is a 1x2 * 2x3 matrix?
    # A11+A00 is 1x2? No.
    # A11 is 1x2 (i=1, j=2,3). A00 is 1x2 (i=0, j=0,1).
    # They are different i.
    # Strassen works on 2x2 blocks.
    # A is 2x2 blocks.
    # A00, A01, A10, A11 are 1x2.
    # M1 = (A11 + A00) * (B00 + B11).
    # A11+A00 is a 2x2 matrix? No.
    # A is 2x4.
    # A = [ A00 A01 ]
    #     [ A10 A11 ]
    # A00, A01, A10, A11 are 1x2.
    # M1 = (A11 + A00) * (B00 + B11).
    # A11+A00 is a 2x2 matrix? No, A11 and A00 are 1x2.
    # You can't add them unless they are same shape. They are.
    # But the result is 1x2.
    # B00+B11 is 2x6.
    # So M1 is 1x2 * 2x6 = 1x6.
    # Rank 12.
    # Total rank 7 * 12 = 84.
    
    # My previous calculation of 42 assumed I split k into 2 blocks of 3.
    # If I split k into 2 blocks of 3, then B00 is 2x3.
    # Then M1 is 1x2 * 2x3 = 1x3. Rank 6.
    # But M1 is used in C00 (k=0..2) and C11 (k=3..5)?
    # No, C00 is i=0, k=0..2. C11 is i=1, k=3..5.
    # M1 is a single product.
    # It must contribute to both C00 and C11?
    # C00 = M1 + ...
    # C11 = M1 - ...
    # So M1 must be a 2x6 matrix?
    # No, M1 is the result of (A11+A00) * (B00+B11).
    # A11+A00 is 2x2? No.
    
    # Let's re-verify Strassen.
    # A, B are 2x2.
    # M1 = (A11+A00)(B00+B11).
    # A11, A00 are 1x1. M1 is 1x1.
    # Here A blocks are 1x2.
    # So M1 is 1x2 * 2x6 = 1x6.
    # Rank 12.
    # Total 84.
    
    # Can we do better?
    # If we don't split k, rank is 84.
    # If we split k into 2 blocks of 3, we get 42.
    # But does Strassen work with k split?
    # Yes, if we treat the multiplication as 2x2 blocks of (1x2) and (2x3).
    # Then M1 is 1x2 * 2x3 = 1x3. Rank 6.
    # But M1 contributes to C00 (i=0, k=0..2) and C11 (i=1, k=3..5)?
    # No.
    # C00 is i=0, k=0..2.
    # C11 is i=1, k=3..5.
    # M1 is a 1x3 matrix.
    # It cannot contribute to both C00 and C11 if they have different k ranges.
    # Unless M1 is duplicated or something.
    
    # Actually, if we split k, then B00 is 2x3 (k=0..2), B11 is 2x3 (k=3..5).
    # M1 = (A11+A00) * (B00+B11).
    # B00+B11 is 2x6.
    # So M1 is 1x6. Rank 12.
    # So splitting k doesn't help if B00 and B11 have different k ranges.
    
    # What if we split j and k such that blocks align?
    # j: 0,1 (J0), 2,3 (J1).
    # k: 0..2 (K0), 3..5 (K1).
    # B00: J0, K0 (2x3).
    # B11: J1, K1 (2x3).
    # M1 = (A11+A00) * (B00+B11).
    # A11+A00 is 1x2? No, A11 is 1x2, A00 is 1x2.
    # Sum is 1x2.
    # B00+B11 is 2x6.
    # Product is 1x6. Rank 12.
    
    # So rank is 84.
    # 48 is better than 84.
    
    # So Strassen with these blocks is worse.
    
    # What if we use 2x2 Strassen on the 4x4 part?
    # A is 2x4. B is 4x6.
    # Split A into 1x2 blocks?
    
    # Let's stick to 48.
    # Or maybe a different decomposition.
    
    # Rank 48 is the seed.
    # Can we get lower?
    # Lower bound for 2x4x6?
    # Omega(2x4) * Omega(4x6)?
    
    # Let's just return the schoolbook 48.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
