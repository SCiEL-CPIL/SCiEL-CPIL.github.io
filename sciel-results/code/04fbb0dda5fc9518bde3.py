import numpy as np

def get_strassen_decomp(N):
    if N == 2:
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M0
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # M1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # M2
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # M3
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        # M4
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        # M5
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = -1; W[3, 5] = 1
        # M6
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    if N == 4:
        U2, V2, W2 = get_strassen_decomp(2)
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((32, rank))
        W = np.zeros((32, rank))
        
        # Define A^{(k)} and B^{(k)} as 2x2 matrices of coefficients for the 4 blocks.
        # A blocks: 0, 1, 2, 3 (00, 01, 10, 11)
        # A^{(k)}_{ij} is a 4-vector.
        
        # M0: X0 = A00+A11. Y0 = B00+B11.
        # X0 is a 2x2 matrix. Its entries are the entries of A00+A11.
        # So X0_{00} = A00_{00}+A11_{00}.
        # This means the coefficient of A00 in X0_{00} is 1, A11 is 1.
        # The local index for A00 in the 2x2 problem is 0.
        # The local index for A11 in the 2x2 problem is 3.
        
        # We need to map the global block indices to the local 2x2 problem indices.
        # The local 2x2 problem has 4 blocks: 00, 01, 10, 11.
        # These correspond to the 4 entries of the 2x2 matrix X_k.
        
        # For M0:
        # X0 = A00+A11.
        # X0_{00} = A00_{00}+A11_{00}
        # X0_{01} = A00_{01}+A11_{01}
        # X0_{10} = A00_{10}+A11_{10}
        # X0_{11} = A00_{11}+A11_{11}
        
        # So, for the local 2x2 problem, the input matrix X0 has entries that are linear combinations of the global blocks.
        # Specifically, the local block 00 (index 0) of X0 is formed from global blocks 0 and 3.
        # The local block 01 (index 1) of X0 is formed from global blocks 0 and 3.
        # The local block 10 (index 2) of X0 is formed from global blocks 0 and 3.
        # The local block 11 (index 3) of X0 is formed from global blocks 0 and 3.
        
        # So, for each local block index l_idx (0..3), we have a 4-vector of coefficients for the global blocks.
        # For M0, X0_coeffs[l_idx] = [1, 0, 0, 1] for all l_idx.
        # Y0_coeffs[l_idx] = [1, 0, 0, 1] for all l_idx.
        
        # For M1: X1 = A10+A11. Y1 = B00.
        # X1_coeffs[l_idx] = [0, 0, 1, 1] for all l_idx.
        # Y1_coeffs[l_idx] = [1, 0, 0, 0] for all l_idx.
        
        # For M2: X2 = A00. Y2 = B01-B11.
        # X2_coeffs[l_idx] = [1, 0, 0, 0]
        # Y2_coeffs[l_idx] = [0, 1, 0, -1]
        
        # For M3: X3 = A11. Y3 = B10-B00.
        # X3_coeffs[l_idx] = [0, 0, 0, 1]
        # Y3_coeffs[l_idx] = [0, 0, 1, -1]
        
        # For M4: X4 = A00+A01. Y4 = B11.
        # X4_coeffs[l_idx] = [1, 1, 0, 0]
        # Y4_coeffs[l_idx] = [0, 0, 0, 1]
        
        # For M5: X5 = A10-A00. Y5 = B00+B01.
        # X5_coeffs[l_idx] = [-1, 0, 1, 0]
        # Y5_coeffs[l_idx] = [1, 1, 0, 0]
        
        # For M6: X6 = A01-A11. Y6 = B10+B11.
        # X6_coeffs[l_idx] = [0, 1, 0, -1]
        # Y6_coeffs[l_idx] = [0, 0, 1, 1]
        
        A_coeffs = [
            [[1, 0, 0, 1]]*4,
            [[0, 0, 1, 1]]*4,
            [[1, 0, 0, 0]]*4,
            [[0, 0, 0, 1]]*4,
            [[1, 1, 0, 0]]*4,
            [[-1, 0, 1, 0]]*4,
            [[0, 1, 0, -1]]*4
        ]
        
        B_coeffs = [
            [[1, 0, 0, 1]]*4,
            [[1, 0, 0, 0]]*4,
            [[0, 1, 0, -1]]*4,
            [[0, 0, 1, -1]]*4,
            [[0, 0, 0, 1]]*4,
            [[1, 1, 0, 0]]*4,
            [[0, 0, 1, 1]]*4
        ]
        
        # W coeffs for C blocks from M_k
        # C00 = M0 - M3 + M6 - M5
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        W_coeffs = [
            [1, 0, 0, -1, 0, -1, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        for k in range(7):
            for l in range(7):
                col = k * 7 + l
                
                # U
                for local_row in range(4):
                    # local_row is the index in the 2x2 problem (0..3)
                    # The global row is formed by the 4 blocks.
                    # U[global_row, col] = sum_{global_block} A_coeffs[k][local_row][global_block] * U2[local_row, l]
                    # Wait, U2[local_row, l] is the coefficient of the local block local_row in the l-th product.
                    # The local block local_row is formed from global blocks via A_coeffs[k][local_row].
                    # So U[global_row, col] = sum_{global_block} A_coeffs[k][local_row][global_block] * U2[local_row, l]
                    # But U2[local_row, l] is a scalar.
                    # And global_row corresponds to a specific entry in the global 4x4 matrix.
                    # The global 4x4 matrix is composed of 4 blocks of size 2x2.
                    # So global_row = block_row * 2 + local_entry_row.
                    # The local block local_row in the 2x2 problem corresponds to a specific block in the 4x4 matrix?
                    # No, the 2x2 problem is on the matrix X_k, which is a 2x2 matrix of 2x2 blocks.
                    # So the local row in the 2x2 problem is the block row index.
                    # The local entry row is the row within the block.
                    
                    # So, for a fixed k, l, and local_row (block row in 2x2 problem):
                    # The contribution to U[global_row, col] comes from the block at local_row.
                    # The block at local_row in X_k is formed from global blocks via A_coeffs[k][local_row].
                    # The U2[local_row, l] is the coefficient of the block at local_row in the l-th product of the 2x2 problem.
                    # So, for each global block gb (0..3) and each local entry row er (0..1):
                    # global_row = local_row * 2 + er? No.
                    # The 2x2 problem has 4 blocks.
                    # The global 4x4 matrix has 4 blocks.
                    # The local block index local_row in the 2x2 problem corresponds to one of the 4 global blocks?
                    # No, X_k is a 2x2 matrix. Its entries are 2x2 blocks.
                    # So the local block index local_row in the 2x2 problem is the (i,j) index of the 2x2 matrix X_k.
                    # But X_k is formed from the 4 global blocks.
                    # So each entry of X_k is a linear combination of the 4 global blocks.
                    # So, for a fixed local_row (which is an entry index 0..3 in the 2x2 matrix),
                    # the coefficient of global block gb in that entry is A_coeffs[k][local_row][gb].
                    # The U2[local_row, l] is the coefficient of that entry in the l-th product.
                    # So the total coefficient of global block gb in the l-th product of the k-th Strassen step is
                    # sum_{local_row} A_coeffs[k][local_row][gb] * U2[local_row, l].
                    
                    # And this applies to all rows within the block gb.
                    # So for each global block gb (0..3) and each local entry row er (0..1):
                    # global_row = gb * 2 + er.
                    # U[global_row, col] = sum_{local_row} A_coeffs[k][local_row][gb] * U2[local_row, l]
                    
                    for gb in range(4):
                        for er in range(2):
                            global_row = gb * 2 + er
                            coeff = 0
                            for local_row in range(4):
                                coeff += A_coeffs[k][local_row][gb] * U2[local_row, l]
                            U[global_row, col] = coeff
                            
                # V
                # Similar for V.
                # V[global_row, col] = sum_{local_row} B_coeffs[k][local_row][gb] * V2[local_row, l]
                # global_row = gb * 2 + er.
                for gb in range(4):
                    for er in range(2):
                        global_row = gb * 2 + er
                        coeff = 0
                        for local_row in range(4):
                            coeff += B_coeffs[k][local_row][gb] * V2[local_row, l]
                        V[global_row, col] = coeff
                        
                # W
                # W[global_row, col] = sum_{k'} W_coeffs[gb][k'] * W2[local_row, l]?
                # No, W is for the output.
                # The output C is a 4x4 matrix of blocks.
                # Each block C_gb is a linear combination of the 7 products M_k.
                # C_gb = sum_{k} W_coeffs[gb][k] M_k.
                # Each M_k is decomposed into 7 products via the 2x2 Strassen.
                # M_k = sum_{l} U2_l ... W2_l.
                # So C_gb = sum_{k} sum_{l} W_coeffs[gb][k] W2[local_row, l] ...
                # The W2[local_row, l] is the coefficient of the l-th product in the k-th Strassen step for the block at local_row.
                # But the block at local_row in the 2x2 problem corresponds to a specific block in the 4x4 output?
                # No, the 2x2 problem output is a 2x2 matrix of 2x2 blocks.
                # So the local block index local_row in the 2x2 problem corresponds to one of the 4 global output blocks.
                # So for a fixed global output block gb (0..3) and local entry row er (0..1):
                # global_row = gb * 2 + er.
                # W[global_row, col] = sum_{local_row} W2[local_row, l] * (coefficient of M_k in C_gb)
                # But M_k is the product for the k-th Strassen step.
                # The output of the k-th Strassen step is a 2x2 matrix of 2x2 blocks.
                # The block at local_row in this output is added to C_gb with coefficient W_coeffs[gb][k].
                # So W[global_row, col] = W_coeffs[gb][k] * W2[local_row, l]
                # But we sum over local_row? No, local_row is the block index in the 2x2 output.
                # And gb is the block index in the 4x4 output.
                # The 2x2 output blocks are mapped to the 4x4 output blocks.
                # The 2x2 output has 4 blocks. The 4x4 output has 4 blocks.
                # So local_row == gb.
                # So W[global_row, col] = W_coeffs[gb][k] * W2[gb, l]
                
                for gb in range(4):
                    for er in range(2):
                        global_row = gb * 2 + er
                        W[global_row, col] = W_coeffs[gb][k] * W2[gb, l]
                        
        return U, V, W

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U49, V49, W49 = get_strassen_decomp(4)
    
    for i in range(16):
        for k in range(49):
            U[i][k] = U49[i, k]
            U[i][k + 49] = U49[i, k]
            
    for i in range(16):
        for k in range(49):
            V[i][k] = V49[i, k]
            V[i + 16][k + 49] = V49[i, k]
            
    for i in range(16):
        for k in range(49):
            W[i][k] = W49[i, k]
            W[i + 16][k + 49] = W49[i, k]
            
    return {"U": U, "V": V, "W": W}
