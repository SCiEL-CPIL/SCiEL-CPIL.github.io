"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    
    # Strassen rank-7 decomposition for 2x2 matrices.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B where C[i,j] = sum_k A[i,k]*B[k,j]
    # 
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)
    # M4 = a11*(b01-b11)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # U vectors (for A, flattened as [a00, a01, a10, a11]):
    U_strassen = [
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [1, 0, -1, 0],  # M6: a10-a00
        [0, 1, 0, -1]   # M7: a01-a11
    ]
    
    # V vectors (for B, flattened as [b00, b01, b10, b11]):
    V_strassen = [
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b00
        [0, 1, 0, -1],  # M4: b01-b11
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00+b01
        [0, 0, 1, 1]    # M7: b10+b11
    ]
    
    # W vectors (for C, flattened as [c00, c01, c10, c11]):
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    W_strassen = [
        [1, 0, 0, 1],   # M1: contributes to C00 and C11
        [0, 0, 1, -1],  # M2: contributes to C10 and C11
        [0, 1, 0, 1],   # M3: contributes to C01 and C11
        [1, 0, 0, 0],   # M4: contributes to C00
        [-1, 1, 0, 0],  # M5: contributes to C00 and C01
        [0, 0, 1, 1],   # M6: contributes to C10 and C11
        [1, 0, 0, 0]    # M7: contributes to C00
    ]
    
    # Verify Strassen decomposition:
    # C00 should be: M1 - M5 + M4 + M7
    # = 1 - (-1) + 1 + 1 = 4? No, let me re-check.
    # M1[0] = 1, M5[0] = -1, M4[0] = 1, M7[0] = 1
    # Sum for C00: 1 + 1 + 1 + 1 = 4? That's wrong.
    
    # Let me re-derive. The W vector for each M should indicate which C elements it contributes to.
    # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # This contributes to C00 (via a00*b00), C01 (via a00*b11), C10 (via a11*b00), C11 (via a11*b11)
    # So M1 W = [1, 1, 1, 1]? No, that's not right either.
    
    # Actually, the Strassen algorithm computes intermediate values M1-M7, and then:
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # So the W vectors should be:
    # M1: [1, 0, 0, 1] (contributes to C00 with +1, C11 with +1)
    # M2: [0, 0, 1, -1] (contributes to C10 with +1, C11 with -1)
    # M3: [0, 1, 0, 1] (contributes to C01 with +1, C11 with +1)
    # M4: [1, 0, 0, 0] (contributes to C00 with +1)
    # M5: [-1, 1, 0, 0] (contributes to C00 with -1, C01 with +1)
    # M6: [0, 0, 1, 1] (contributes to C10 with +1, C11 with +1)
    # M7: [1, 0, 0, 0] (contributes to C00 with +1)
    
    # Let me verify C00: M1[0] + M4[0] - M5[0] + M7[0] = 1 + 1 - (-1) + 1 = 4? 
    # Wait, C00 = M1 - M5 + M4 + M7, so it's M1[0] + (-1)*M5[0] + M4[0] + M7[0]
    # = 1 + (-1)*(-1) + 1 + 1 = 1 + 1 + 1 + 1 = 4? That's still wrong.
    
    # I think I'm confusing myself. Let me just directly construct the decomposition for 4x4x6
    # using 11 Strassen instances.
    
    # For 4x4x6, we can split into 2x2 blocks:
    # A is 2x2 blocks of 2x2 matrices
    # B is 2x3 blocks of 2x2 matrices  
    # C is 2x3 blocks of 2x2 matrices
    
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2}
    
    # Each C[i,k] is a 2x2 matrix = sum of 2 products of 2x2 matrices
    # We can use Strassen for each product, but that gives 6*2*7 = 84 rank
    
    # Alternatively, we can view this differently. Let me just use the known result
    # that 4x4x6 can be done with rank 77 using 11 Strassen instances.
    
    # Let me construct it by considering the tensor structure.
    # The 4x4x6 multiplication tensor T has T[i,j,k] = 1 if k corresponds to the output.
    # In the trace convention, output index = k*n + i where i is row, k is col.
    # So for C of shape 4x6, output index = col*4 + row.
    
    # Let me just build a rank-77 decomposition by combining Strassen blocks.
    # I'll use the approach where we have 11 independent 2x2x2 Strassen multiplications.
    
    rank = 77
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16
    V = [[0] * rank for _ in range(24)]  # 4*6 = 24
    W = [[0] * rank for _ in range(24)]  # 6*4 = 24
    
    # Define the 11 Strassen instances. Each instance operates on specific
    # 2x2 sub-blocks of A, B, and contributes to specific 2x2 sub-blocks of C.
    
    # Instance definitions: (row_block_A, col_block_A, row_block_B, col_block_B, row_block_C, col_block_C)
    # Where row_block and col_block are 0 or 1 for 2x2 blocks.
    
    # For 4x4x6 with 2x2 blocks:
    # A has blocks A[0,0], A[0,1], A[1,0], A[1,1]
    # B has blocks B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2]
    # C has blocks C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So we have 6 block multiplications, each involving 2 terms.
    # Total 12 individual 2x2*2x2 products, but we can share Strassen computations.
    
    # Let me just use 11 Strassen instances with appropriate projections.
    
    # Instance 1: A[0,0] * B[0,0] -> C[0,0]
    # Instance 2: A[0,1] * B[1,0] -> C[0,0]
    # Instance 3: A[0,0] * B[0,1] -> C[0,1]
    # Instance 4: A[0,1] * B[1,1] -> C[0,1]
    # Instance 5: A[0,0] * B[0,2] -> C[0,2]
    # Instance 6: A[0,1] * B[1,2] -> C[0,2]
    # Instance 7: A[1,0] * B[0,0] -> C[1,0]
    # Instance 8: A[1,1] * B[1,0] -> C[1,0]
    # Instance 9: A[1,0] * B[0,1] -> C[1,1]
    # Instance 10: A[1,1] * B[1,1] -> C[1,1]
    # Instance 11: A[1,0] * B[0,2] -> C[1,2]
    # Plus we need A[1,1] * B[1,2] -> C[1,2]
    
    # That's 12 instances, giving rank 84. To get 77, we need to share some computations.
    
    # Let me just implement the straightforward 12-instance version with rank 84 first,
    # then see if we can optimize to 77.
    
    # Actually, let me just return a valid decomposition with rank 77.
    # I'll construct it carefully.
    
    # Helper function to add a Strassen instance
    def add_strassen_instance(base_col, a_row_block, a_col_block, b_row_block, b_col_block, c_row_block, c_col_block):
        """Add one Strassen instance at the given block positions."""
        for s in range(7):
            col = base_col + s
            # U: extract from A[a_row_block*2 + i, a_col_block*2 + j]
            for local_i in range(2):
                for local_j in range(2):
                    u_idx = (a_row_block * 2 + local_i) * 4 + (a_col_block * 2 + local_j)
                    # U_strassen[s] is [u00, u01, u10, u11]
                    U[u_idx][col] = U_strassen[s][local_i * 2 + local_j]
            
            # V: extract from B[b_row_block*2 + i, b_col_block*2 + j]
            for local_i in range(2):
                for local_j in range(2):
                    v_idx = (b_row_block * 2 + local_i) * 6 + (b_col_block * 2 + local_j)
                    V[v_idx][col] = V_strassen[s][local_i * 2 + local_j]
            
            # W: contribute to C[c_row_block*2 + i, c_col_block*2 + j]
            # In trace convention, C index = col * n + row = col * 4 + row
            for local_i in range(2):
                for local_j in range(2):
                    w_idx = (c_col_block * 2 + local_j) * 4 + (c_row_block * 2 + local_i)
                    W[w_idx][col] = W_strassen[s][local_i * 2 + local_j]
    
    # Add 11 Strassen instances
    instances = [
        # (a_row, a_col, b_row, b_col, c_row, c_col)
        (0, 0, 0, 0, 0, 0),  # 1
        (0, 1, 1, 0, 0, 0),  # 2
        (0, 0, 0, 1, 0, 1),  # 3
        (0, 1, 1, 1, 0, 1),  # 4
        (0, 0, 0, 2, 0, 2),  # 5
        (0, 1, 1, 2, 0, 2),  # 6
        (1, 0, 0, 0, 1, 0),  # 7
        (1, 1, 1, 0, 1, 0),  # 8
        (1, 0, 0, 1, 1, 1),  # 9
        (1, 1, 1, 1, 1, 1),  # 10
        (1, 0, 0, 2, 1, 2),  # 11
    ]
    
    for idx, (ar, ac, br, bc, cr, cc) in enumerate(instances):
        add_strassen_instance(idx * 7, ar, ac, br, bc, cr, cc)
    
    # We have 11 instances * 7 = 77 columns used.
    # But we're missing A[1,1] * B[1,2] -> C[1,2]
    # This means the decomposition is incomplete!
    
    # I need to include all 12 products. Let me reconsider.
    # Maybe I can merge some instances.
    
    # For now, let me just return the schoolbook with rank 96 since I'm not sure
    # about the exact 77-rank construction.
    
    rank = 4 * 4 * 6
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U[i * 4 + j][column] = 1
                V[j * 6 + k][column] = 1
                W[k * 4 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
