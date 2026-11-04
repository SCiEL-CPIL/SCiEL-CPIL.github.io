import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Part 1: 4x4 * 4x4 (cols 0-3 of B)
    # Rank 49 using Strassen
    # Part 2: 4x4 * 4x3 (cols 4-6 of B)
    # Rank 48 using Schoolbook
    
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((28, total_rank), dtype=float)
    W = np.zeros((28, total_rank), dtype=float)
    
    # --- Part 1: Strassen for 4x4x4 ---
    # We need the Strassen decomposition for 4x4x4.
    # This is 7 applications of Strassen for 2x2x2.
    # Strassen for 2x2x2 has rank 7.
    # The 7 products are:
    # M1 = A11 * (B22 - B12)
    # M2 = (A11 + A21) * B12
    # M3 = (A12 + A22) * B21
    # M4 = A22 * (B11 - B21)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 - A21) * (B11 + B12)
    
    # For 4x4x4, we apply this to 2x2 blocks.
    # Let A be blocks A11, A12, A21, A22 (each 2x2).
    # Let B be blocks B11, B12, B21, B22 (each 2x2).
    # The 7 products M1..M7 are 2x2 matrices.
    # Each Mk is computed by 7 scalar multiplications (rank 7 decomposition of 2x2x2).
    
    # We need to map the 2x2x2 Strassen terms to the global 4x4x4 indices.
    
    # Indices for A (16):
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Indices for B (16 for part 1):
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # Indices for C (16 for part 1):
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # Strassen 2x2x2 decomposition (rank 7)
    # u, v, w vectors of length 4.
    # The 7 terms are:
    # 1: u=(1,0,0,-1), v=(0,0,0,1), w=(0,0,0,1) ? No.
    # Let's use the standard Strassen formulas.
    # M1 = A11 * (B22 - B12)
    # M2 = (A11 + A21) * B12
    # M3 = (A12 + A22) * B21
    # M4 = A22 * (B11 - B21)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 - A21) * (B11 + B12)
    
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M5 - M1 + M3 - M7
    
    # For each Mk, we have a 2x2x2 multiplication.
    # The rank 7 decomposition of 2x2x2 is exactly these 7 products?
    # No, the 7 products ARE the decomposition.
    # Each product is a scalar multiplication of two linear combinations.
    # So for each Mk, we have 1 term?
    # No, Mk is a 2x2 matrix.
    # Mk = Xk * Yk where Xk, Yk are 2x2 matrices.
    # The multiplication of 2x2 matrices has rank 7.
    # So each Mk requires 7 terms.
    # Total 49 terms.
    
    # We need the rank 7 decomposition of M_{2,2,2}.
    # This is the same as the Strassen algorithm for 2x2 matrices.
    # The 7 terms are the 7 products listed above.
    # Wait, the 7 products listed above ARE the Strassen algorithm for 2x2 matrices.
    # So for the 4x4 case, we have 7 "super-products" M1..M7.
    # Each Mk is a 2x2 matrix multiplication.
    # To compute Mk, we use the Strassen algorithm again?
    # No, Mk is a 2x2 matrix. We can just use schoolbook for 2x2?
    # Rank of 2x2x2 is 7.
    # So we use the rank 7 decomposition for each Mk.
    
    # Let's define the rank 7 decomposition of M_{2,2,2}.
    # T = sum_{s=1}^7 u_s \otimes v_s \otimes w_s
    # The vectors u_s, v_s, w_s are of length 4.
    # We can generate them from the Strassen formulas.
    
    # Strassen 2x2x2 terms:
    # s=1: M1 = A11 * (B22 - B12)
    # u1 = [1, 0, 0, 0] (A11)
    # v1 = [0, 0, -1, 1] (B22 - B12) -> indices 0,1,2,3 for B11,B12,B21,B22?
    # Let's fix indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # A11 is at 0. A12 is at 1. A21 is at 2. A22 is at 3.
    # B11 is at 0. B12 is at 1. B21 is at 2. B22 is at 3.
    
    # M1 = A11 * (B22 - B12)
    # u1 = [1, 0, 0, 0]
    # v1 = [0, -1, 0, 1]
    # w1 = [0, 1, 0, 0] ? No, w is for C.
    # C12 = M1 + M2.
    # So M1 contributes to C12.
    # C12 is at index 1.
    # So w1 = [0, 1, 0, 0].
    
    # M2 = (A11 + A21) * B12
    # u2 = [1, 0, 1, 0]
    # v2 = [0, 1, 0, 0]
    # w2 = [0, 1, 0, 0]
    
    # M3 = (A12 + A22) * B21
    # u3 = [0, 1, 0, 1]
    # v3 = [0, 0, 1, 0]
    # w3 = [0, 0, 1, 0]
    
    # M4 = A22 * (B11 - B21)
    # u4 = [0, 0, 0, 1]
    # v4 = [1, 0, -1, 0]
    # w4 = [1, 0, 1, 0]
    
    # M5 = (A11 + A22) * (B11 + B22)
    # u5 = [1, 0, 0, 1]
    # v5 = [1, 0, 0, 1]
    # w5 = [1, 0, 0, 1]
    
    # M6 = (A12 - A22) * (B21 + B22)
    # u6 = [0, 1, 0, -1]
    # v6 = [0, 0, 1, 1]
    # w6 = [0, 0, 0, 1]
    
    # M7 = (A11 - A21) * (B11 + B12)
    # u7 = [1, 0, -1, 0]
    # v7 = [1, 1, 0, 0]
    # w7 = [0, 0, 0, -1]
    
    # Check C11 = M5 + M4 - M2 + M6
    # w11 = w5 + w4 - w2 + w6
    # = [1,0,0,1] + [1,0,1,0] - [0,1,0,0] + [0,0,0,1]
    # = [2, -1, 1, 2]
    # But C11 should be at index 0.
    # The formula is C11 = M5 + M4 - M2 + M6.
    # The w vectors are the coefficients for the output.
    # So for term s, the contribution to C is w_s * (u_s . A) * (v_s . B).
    # Summing over s gives C.
    
    # For the 4x4x4 case, we have 7 such blocks.
    # Block k corresponds to the k-th Strassen product of the 4x4 matrix.
    # For each block k, we have 7 terms (from the 2x2x2 decomposition).
    
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    # Let's implement this carefully.
    
    # Local 2x2 indices: 0,1,2,3
    # Global 4x4 indices for block (bi, bj):
    # bi, bj in {0, 1}
    # Local index l corresponds to global index (bi*2 + l//2)*4 + (bj*2 + l%2)
    
    # Strassen 4x4x4 blocks:
    # A11: bi=0, bj=0
    # A12: bi=0, bj=1
    # A21: bi=1, bj=0
    # A22: bi=1, bj=1
    
    # B11: bi=0, bj=0
    # B12: bi=0, bj=1
    # B21: bi=1, bj=0
    # B22: bi=1, bj=1
    
    # C11: bi=0, bj=0
    # C12: bi=0, bj=1
    # C21: bi=1, bj=0
    # C22: bi=1, bj=1
    
    # The 7 Strassen products for 4x4:
    # M1 = A11 * (B22 - B12)
    # M2 = (A11 + A21) * B12
    # M3 = (A12 + A22) * B21
    # M4 = A22 * (B11 - B21)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 - A21) * (B11 + B12)
    
    # For each Mk, we compute a 2x2 matrix.
    # We use the 7-term decomposition for 2x2x2.
    
    # Let's build the U, V, W for Part 1.
    
    col = 0
    for k in range(7):
        # Determine the linear combinations for A, B, C for this Strassen product Mk
        # We need the u, v, w vectors for the 4x4 level.
        # But we are decomposing Mk (2x2x2) into 7 scalar products.
        # So we iterate over the 7 local terms.
        
        # Local u, v, w for the 7 terms of 2x2x2
        local_u = [
            [1, 0, 0, 0], # M1
            [1, 0, 1, 0], # M2
            [0, 1, 0, 1], # M3
            [0, 0, 0, 1], # M4
            [1, 0, 0, 1], # M5
            [0, 1, 0, -1], # M6
            [1, 0, -1, 0]  # M7
        ]
        local_v = [
            [0, 0, 0, 1], # M1: B22 - B12 -> [0, -1, 0, 1] ?
            # Wait, my local_v above was for the 4x4 level?
            # No, local_v is for the 2x2 B matrix.
            # B11=0, B12=1, B21=2, B22=3.
            # M1 = A11 * (B22 - B12)
            # v1 = [0, -1, 0, 1]
            [0, -1, 0, 1],
            [0, 1, 0, 0], # M2: B12
            [0, 0, 1, 0], # M3: B21
            [1, 0, -1, 0], # M4: B11 - B21
            [1, 0, 0, 1], # M5: B11 + B22
            [0, 0, 1, 1], # M6: B21 + B22
            [1, 1, 0, 0]  # M7: B11 + B12
        ]
        local_w = [
            [0, 1, 0, 0], # M1 -> C12
            [0, 1, 0, 0], # M2 -> C12
            [0, 0, 1, 0], # M3 -> C21
            [1, 0, 1, 0], # M4 -> C11, C21
            [1, 0, 0, 1], # M5 -> C11, C22
            [0, 0, 0, 1], # M6 -> C22
            [0, 0, 0, -1] # M7 -> C22
        ]
        
        # Now, for the 4x4 level, Mk is a product of two 2x2 matrices.
        # The "A" part of Mk is a linear combination of the 4 blocks of A.
        # The "B" part of Mk is a linear combination of the 4 blocks of B.
        # The "C" part of Mk is a linear combination of the 4 blocks of C.
        
        # We need to map the local 2x2 indices to the 4x4 block indices.
        # For a given local term s in the 2x2x2 decomposition of Mk:
        # u_global is a vector of length 16.
        # It is formed by taking the local_u[s] vector (length 4) and mapping it to the 4 blocks of A.
        # But wait, the local_u[s] corresponds to the entries of the 2x2 matrix A_mk.
        # A_mk is a linear combination of the 4 blocks of A.
        # So u_global will have non-zero entries in the blocks specified by the Strassen formula for Mk.
        
        # Let's define the Strassen formulas for 4x4x4 explicitly.
        # A11: block (0,0)
        # A12: block (0,1)
        # A21: block (1,0)
        # A22: block (1,1)
        
        # M1 = A11 * (B22 - B12)
        # A part: A11
        # B part: B22 - B12
        # C part: C12
        
        # M2 = (A11 + A21) * B12
        # A part: A11 + A21
        # B part: B12
        # C part: C12
        
        # M3 = (A12 + A22) * B21
        # A part: A12 + A22
        # B part: B21
        # C part: C21
        
        # M4 = A22 * (B11 - B21)
        # A part: A22
        # B part: B11 - B21
        # C part: C11 + C21
        
        # M5 = (A11 + A22) * (B11 + B22)
        # A part: A11 + A22
        # B part: B11 + B22
        # C part: C11 + C22
        
        # M6 = (A12 - A22) * (B21 + B22)
        # A part: A12 - A22
        # B part: B21 + B22
        # C part: C22
        
        # M7 = (A11 - A21) * (B11 + B12)
        # A part: A11 - A21
        # B part: B11 + B12
        # C part: C22 (with -1)
        
        # For each Mk, we have a 2x2x2 multiplication.
        # We decompose this 2x2x2 multiplication into 7 scalar products.
        # For each scalar product s:
        # u_global = (A_part_of_Mk) * local_u[s]
        # v_global = (B_part_of_Mk) * local_v[s]
        # w_global = (C_part_of_Mk) * local_w[s]
        
        # We need to define the A_part, B_part, C_part for each Mk.
        # These are vectors of length 4 (for the 4 blocks).
        
        A_parts = [
            [1, 0, 0, 0], # M1: A11
            [1, 0, 1, 0], # M2: A11 + A21
            [0, 1, 0, 1], # M3: A12 + A22
            [0, 0, 0, 1], # M4: A22
            [1, 0, 0, 1], # M5: A11 + A22
            [0, 1, 0, -1], # M6: A12 - A22
            [1, 0, -1, 0]  # M7: A11 - A21
        ]
        B_parts = [
            [0, 0, 0, 1], # M1: B22 - B12 -> [0, -1, 0, 1]
            [0, -1, 0, 1],
            [0, 1, 0, 0], # M2: B12
            [0, 0, 1, 0], # M3: B21
            [1, 0, -1, 0], # M4: B11 - B21
            [1, 0, 0, 1], # M5: B11 + B22
            [0, 0, 1, 1], # M6: B21 + B22
            [1, 1, 0, 0]  # M7: B11 + B12
        ]
        C_parts = [
            [0, 1, 0, 0], # M1: C12
            [0, 1, 0, 0], # M2: C12
            [0, 0, 1, 0], # M3: C21
            [1, 0, 1, 0], # M4: C11 + C21
            [1, 0, 0, 1], # M5: C11 + C22
            [0, 0, 0, 1], # M6: C22
            [0, 0, 0, -1] # M7: -C22
        ]
        
        for s in range(7):
            # Construct U, V, W for this term
            u_global = np.zeros(16)
            v_global = np.zeros(16) # Only first 16 of 28
            w_global = np.zeros(16) # Only first 16 of 28
            
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff_a = A_parts[k][block_idx]
                    coeff_b = B_parts[k][block_idx]
                    coeff_c = C_parts[k][block_idx]
                    
                    if coeff_a != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                global_idx = (bi * 2 + li) * 4 + (bj * 2 + lj)
                                u_global[global_idx] = coeff_a * local_u[s][local_idx]
                                
                    if coeff_b != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                global_idx = (bi * 2 + li) * 4 + (bj * 2 + lj)
                                v_global[global_idx] = coeff_b * local_v[s][local_idx]
                                
                    if coeff_c != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                global_idx = (bi * 2 + li) * 4 + (bj * 2 + lj)
                                w_global[global_idx] = coeff_c * local_w[s][local_idx]
                                
            U[:, col] = u_global
            V[:, col] = v_global
            W[:, col] = w_global
            col += 1
            
    # --- Part 2: Schoolbook for 4x4x3 ---
    # Cols 4,5,6 of B.
    # Rank 48.
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # A[i,j] * B[j, 4+k]
                U[i * 4 + j, col] = 1
                V[j * 7 + (4 + k), col] = 1
                W[(4 + k) * 4 + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
