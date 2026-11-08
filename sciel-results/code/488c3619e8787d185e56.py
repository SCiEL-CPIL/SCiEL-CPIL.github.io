import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate Strassen decomposition for 4x4x4 (Rank 49)
    def get_strassen_matrices(size):
        if size == 1:
            return [[1]], [[1]], [[1]]
        
        half = size // 2
        U_half, V_half, W_half = get_strassen_matrices(half)
        
        # Indices for size x size matrices: 0..size*size-1
        # Block indices: 00, 01, 10, 11 corresponding to (0,0), (0,1), (1,0), (1,1)
        # Global index for element (r, c) is r*size + c
        
        # The 7 Strassen multiplications
        # A_k = sum coeff * A_block
        # B_k = sum coeff * B_block
        # C_block = sum coeff * P_k
        
        # Coefficients for A blocks (2x2 grid of blocks)
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # A coefficients for each of the 7 steps
        # A_blocks: 00, 01, 10, 11
        A_coeffs = [
            [1, 0, 0, 1], # M1: A00 + A11
            [0, 0, 1, 1], # M2: A10 + A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00 + A01
            [0, 0, 1, -1],# M6: A10 - A00
            [0, 1, 0, -1] # M7: A01 - A11
        ]
        
        B_coeffs = [
            [1, 0, 0, 1], # M1: B00 + B11
            [1, 0, 0, 0], # M2: B00
            [0, 1, 0, -1],# M3: B01 - B11
            [0, 0, 1, -1],# M4: B10 - B00
            [0, 0, 0, 1], # M5: B11
            [1, 1, 0, 0], # M6: B00 + B01
            [0, 0, 1, 1]  # M7: B10 + B11
        ]
        
        # Result coefficients for C blocks (2x2 grid)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        C_coeffs = [
            [1, 0, 0, 1, 0, 0, 0],  # M1
            [0, -1, 0, 0, 0, 0, 0], # M2
            [0, 0, 1, 0, 0, 0, 0],  # M3
            [0, 0, 0, 1, 0, 0, 0],  # M4
            [0, 0, 0, 0, 1, 0, 0],  # M5
            [0, 0, 0, 0, 0, 1, 0],  # M6
            [0, 0, 0, 0, 0, 0, 1]   # M7
        ]
        # Wait, C_coeffs should map C_blocks to M's.
        # C00: M1 + M4 - M5 + M7 -> [1, 0, 0, 1, -1, 0, 1]
        # C01: M3 + M5 -> [0, 0, 1, 0, 1, 0, 0]
        # C10: M2 + M4 -> [0, 1, 0, 1, 0, 0, 0]
        # C11: M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0]
        
        C_block_coeffs = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 0, 1, 0, 1, 0, 0],  # C01
            [0, 1, 0, 1, 0, 0, 0],  # C10
            [1, -1, 1, 0, 0, 1, 0]  # C11
        ]

        rank = 7 * (half * half)
        U = [[0] * rank for _ in range(size * size)]
        V = [[0] * rank for _ in range(size * size)]
        W = [[0] * rank for _ in range(size * size)]
        
        # Map global indices to block indices
        # Block (br, bc) contains elements (i, j) where i = br*half + r, j = bc*half + c
        # Global index = i * size + j
        
        for step in range(7):
            # Offset for this step's sub-matrices
            offset = step * (half * half)
            
            # Fill U
            for br in range(2):
                for bc in range(2):
                    coeff = A_coeffs[step][br * 2 + bc]
                    if coeff == 0:
                        continue
                    # Iterate over elements in this block
                    for r in range(half):
                        for c in range(half):
                            i = br * half + r
                            j = bc * half + c
                            u_idx = i * size + j
                            # The sub-problem U_half row corresponds to (r, c) in the half-size matrix
                            # u_half_idx = r * half + c
                            u_half_idx = r * half + c
                            for k in range(half * half):
                                U[u_idx][offset + k] += coeff * U_half[u_half_idx][k]
            
            # Fill V
            for br in range(2):
                for bc in range(2):
                    coeff = B_coeffs[step][br * 2 + bc]
                    if coeff == 0:
                        continue
                    for r in range(half):
                        for c in range(half):
                            i = br * half + r
                            j = bc * half + c
                            v_idx = i * size + j
                            v_half_idx = r * half + c
                            for k in range(half * half):
                                V[v_idx][offset + k] += coeff * V_half[v_half_idx][k]
            
            # Fill W
            # W_half maps result elements of the sub-product to the global W
            # The sub-product P_step is a half x half matrix.
            # Its elements contribute to the global C blocks.
            # C_block (br, bc) gets contribution coeff * P_step
            for br in range(2):
                for bc in range(2):
                    coeff = C_block_coeffs[br * 2 + bc][step]
                    if coeff == 0:
                        continue
                    for r in range(half):
                        for c in range(half):
                            i = br * half + r
                            j = bc * half + c
                            w_idx = i * size + j
                            w_half_idx = r * half + c
                            for k in range(half * half):
                                W[w_idx][offset + k] += coeff * W_half[w_half_idx][k]
                                
        return U, V, W

    U4, V4, W4 = get_strassen_matrices(4)
    rank_4 = len(U4[0]) # 49
    
    # 2. Construct full 4x4x7 matrices
    # U: 16 x (49 + 48)
    # V: 28 x (49 + 48)
    # W: 28 x (49 + 48)
    
    total_rank = rank_4 + 48
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Copy 4x4x4 part
    for i in range(16):
        for k in range(rank_4):
            U[i][k] = U4[i][k]
            # V4 rows 0..15 map to V rows 0..15 (B[:, 0:4])
            V[i][k] = V4[i][k]
            # W4 rows 0..15 map to W rows 0..15 (C[:, 0:4])
            W[i][k] = W4[i][k]
    
    # 3. Add 4x4x3 part (Schoolbook for 3 columns)
    # B[:, 4:7] -> V rows 16..27
    # C[:, 4:7] -> W rows 16..27
    col_offset = rank_4
    for c in range(3): # 3 columns
        for r_A in range(4): # Rows of A
            for c_B in range(4): # Cols of A (which are rows of B block)
                # U index: r_A * 4 + c_B
                u_idx = r_A * 4 + c_B
                # V index: c_B * 7 + (4 + c)
                v_idx = c_B * 7 + (4 + c)
                # W index: r_A * 4 + (4 + c)
                w_idx = r_A * 4 + (4 + c)
                
                U[u_idx][col_offset] = 1
                V[v_idx][col_offset] = 1
                W[w_idx][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
