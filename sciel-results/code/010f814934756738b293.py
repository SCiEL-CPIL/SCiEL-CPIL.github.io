import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # We split the 6 columns into 4 and 2.
    # The first 4 columns correspond to p1=4.
    # The last 2 columns correspond to p2=2.
    
    # --- Strassen 4x4x4 ---
    # Strassen 2x2x2 base case
    # Inputs: A (2x2), B (2x2)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to build the U, V, W matrices for the 4x4x4 case.
    # This is done by recursively applying the 2x2x2 Strassen scheme.
    # Since the code needs to be explicit and fast, I will hardcode the generation logic.
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2, rank 7
        # U maps A (2x2=4) to rank 7
        # V maps B (2x2=4) to rank 7
        # W maps C (2x2=4) to rank 7
        # Indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
        # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A11+A22)(B11+B22)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (A21+A22)B11
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # M3 = A11(B21-B22)
        U[0,2] = 1
        V[2,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # M4 = A22(B12-B11)
        U[3,3] = 1
        V[1,3] = 1; V[0,3] = -1
        W[1,3] = 1; W[2,3] = 1
        
        # M5 = (A11+A12)B22
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[3,4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[2,6] = -1
        
        return U, V, W

    # Build 4x4x4 Strassen
    # A is 4x4. Split into 2x2 blocks: A11, A12, A21, A22.
    # B is 4x4. Split into 2x2 blocks: B11, B12, B21, B22.
    # C is 4x4. Split into 2x2 blocks: C11, C12, C21, C22.
    
    # We apply Strassen 2x2x2 to the "super-elements" which are 2x2 matrices.
    # This creates 7 intermediate 2x2x2 multiplications.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # We need to construct the global U, V, W.
    # U_global (16, 49)
    # V_global (16, 49)
    # W_global (16, 49)
    
    # The indices of the 4x4 matrix A (flattened 0..15)
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # We define the mapping for the 7 Strassen steps.
    # Step k (0..6) corresponds to one of the M_k in the 2x2x2 formula.
    # For each step k, we need to form the linear combination of A's blocks and B's blocks.
    # Then we multiply the resulting 2x2 matrices using the base Strassen (rank 7).
    
    U_44 = np.zeros((16, 49), dtype=float)
    V_44 = np.zeros((16, 49), dtype=float)
    W_44 = np.zeros((16, 49), dtype=float)
    
    U_base, V_base, W_base = get_strassen_2x2x2()
    
    # Define block indices for A and B
    # A blocks: 0:(0,1,4,5), 1:(2,3,6,7), 2:(8,9,12,13), 3:(10,11,14,15)
    # B blocks: same indices
    # C blocks: same indices
    
    # Strassen coefficients for A (M1..M7)
    # M1: A11+A22 -> Blocks 0, 3
    # M2: A21+A22 -> Blocks 2, 3
    # M3: A11 -> Block 0
    # M4: A22 -> Block 3
    # M5: A11+A12 -> Blocks 0, 1
    # M6: A21-A11 -> Blocks 2, 0
    # M7: A12-A22 -> Blocks 1, 3
    
    A_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    # Strassen coefficients for B (M1..M7)
    # M1: B11+B22 -> Blocks 0, 3
    # M2: B11 -> Block 0
    # M3: B21-B22 -> Blocks 2, 3
    # M4: B12-B11 -> Blocks 1, 0
    # M5: B22 -> Block 3
    # M6: B11+B12 -> Blocks 0, 1
    # M7: B21+B22 -> Blocks 2, 3
    
    B_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {2: 1, 3: -1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    # Strassen coefficients for C (M1..M7)
    # C11: M1+M4-M5+M7 -> Blocks 0
    # C12: M3+M5 -> Blocks 1
    # C21: M2+M4 -> Blocks 2
    # C22: M1-M2+M3+M6 -> Blocks 3
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # M1, M4, M5, M7 contribute to C11? 
        # Wait, the output C is assembled from the 7 intermediate 2x2 results.
        # Let R_k be the 2x2 result of the k-th Strassen step.
        # C11 = R1 + R4 - R5 + R7
        # C12 = R3 + R5
        # C21 = R2 + R4
        # C22 = R1 - R2 + R3 + R6
        
        # So for C11 (Block 0 of C), we sum contributions from R1, R4, R5, R7.
        # For C12 (Block 1), from R3, R5.
        # For C21 (Block 2), from R2, R4.
        # For C22 (Block 3), from R1, R2, R3, R6.
        
        {0: 1, 3: 1, 4: -1, 6: 1}, # C11
        {2: 1, 4: 1},               # C12
        {1: 1, 3: 1},               # C21
        {0: 1, 1: -1, 2: 1, 5: 1}   # C22
    ]
    
    # Construct U_44, V_44, W_44
    for k in range(7):
        # U part
        for block_idx, coeff in A_coeffs[k].items():
            # Map block_idx to 4 indices in A
            if block_idx == 0: indices = [0, 1, 4, 5]
            elif block_idx == 1: indices = [2, 3, 6, 7]
            elif block_idx == 2: indices = [8, 9, 12, 13]
            else: indices = [10, 11, 14, 15]
            
            for i, idx in enumerate(indices):
                U_44[idx, k*7 + i] = coeff * U_base[i, 0] # Wait, U_base is 4x7.
                # The base Strassen U maps the 4 elements of the 2x2 input to 7 values.
                # Here, the "input" to the k-th step is a 2x2 matrix formed by linear combos of A's blocks.
                # So U_44 should map the global A indices to the 7 slots of the k-th step.
                # The k-th step uses U_base.
                # So U_44[global_idx, k*7 + local_slot] = coeff * U_base[local_idx_in_block, local_slot]
                
        # Let's refine the loop
        for block_idx, coeff in A_coeffs[k].items():
            if block_idx == 0: indices = [0, 1, 4, 5]
            elif block_idx == 1: indices = [2, 3, 6, 7]
            elif block_idx == 2: indices = [8, 9, 12, 13]
            else: indices = [10, 11, 14, 15]
            
            for local_i, global_i in enumerate(indices):
                for local_j in range(7):
                    U_44[global_i, k*7 + local_j] += coeff * U_base[local_i, local_j]
                    
        for block_idx, coeff in B_coeffs[k].items():
            if block_idx == 0: indices = [0, 1, 4, 5]
            elif block_idx == 1: indices = [2, 3, 6, 7]
            elif block_idx == 2: indices = [8, 9, 12, 13]
            else: indices = [10, 11, 14, 15]
            
            for local_i, global_i in enumerate(indices):
                for local_j in range(7):
                    V_44[global_i, k*7 + local_j] += coeff * V_base[local_i, local_j]
                    
        # W part is different. W maps the 7 intermediate results to the final C.
        # The intermediate results are 2x2 matrices.
        # R_k is a 2x2 matrix. Its elements are C_k[0,0], C_k[0,1], C_k[1,0], C_k[1,1].
        # These correspond to the 4 outputs of the k-th base Strassen step.
        # The base Strassen W_base maps the 7 scalar products to the 4 elements of the 2x2 result.
        # So W_44[global_c_idx, k*7 + local_j] should sum the contributions from W_base.
        
        # For each block of C (0..3), we sum the contributions from the R_k's.
        for c_block_idx, coeff_dict in enumerate(C_coeffs):
            # coeff_dict maps k (0..6) to coefficient
            # Wait, C_coeffs is defined as list of dicts for C11..C22.
            # Let's re-structure C_coeffs to map (c_block, k) -> coeff
            
            pass

    # Re-do C_coeffs mapping
    # C11 (block 0) gets R1(1), R4(1), R5(-1), R7(1)
    # C12 (block 1) gets R3(1), R5(1)
    # C21 (block 2) gets R2(1), R4(1)
    # C22 (block 3) gets R1(1), R2(-1), R3(1), R6(1)
    
    C_map = {
        0: {0: 1, 3: 1, 4: -1, 6: 1},
        1: {2: 1, 4: 1},
        2: {1: 1, 3: 1},
        3: {0: 1, 1: -1, 2: 1, 5: 1}
    }
    
    for c_block_idx in range(4):
        if c_block_idx == 0: indices = [0, 1, 4, 5]
        elif c_block_idx == 1: indices = [2, 3, 6, 7]
        elif c_block_idx == 2: indices = [8, 9, 12, 13]
        else: indices = [10, 11, 14, 15]
        
        for local_i, global_i in enumerate(indices):
            for k, coeff in C_map[c_block_idx].items():
                for local_j in range(7):
                    W_44[global_i, k*7 + local_j] += coeff * W_base[local_i, local_j]
                    
    # Part 2: 4x4x2 Schoolbook
    # n=4, m=4, p=2
    # Rank = 32
    U_42 = np.zeros((16, 32), dtype=float)
    V_42 = np.zeros((8, 32), dtype=float)
    W_42 = np.zeros((8, 32), dtype=float)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_42[i*4+j, col] = 1
                V_42[j*2+k, col] = 1
                W_42[k*4+i, col] = 1
                col += 1
                
    # Combine
    U = np.hstack([U_44, U_42])
    V = np.hstack([V_44, V_42])
    W = np.hstack([W_44, W_42])
    
    # Convert to lists of lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
