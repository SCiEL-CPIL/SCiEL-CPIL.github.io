import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 65: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1)
    target_rank = 65
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for 4x4x4 (first 4 columns of C) ---
    # Base Strassen decomposition for 2x2x2
    # Returns list of 7 tuples (u_vec, v_vec, w_vec)
    # u_vec, v_vec, w_vec are length 4 lists (flattened 2x2 matrices)
    def get_base_strassen():
        # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        terms = []
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 0] # C00 += M1
        terms.append((u, v, w))
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, 0] # C10 += M2
        terms.append((u, v, w))
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 0] # C01 += M3
        terms.append((u, v, w))
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 0, 0] # C00 += M4 (Wait, C00 = M1 + M4 - M5 + M7)
        # Correction: C00 gets +M4. C10 gets +M4.
        # Let's trace C00: M1 + M4 - M5 + M7
        # C01: M3 + M5
        # C10: M2 + M4
        # C11: M1 - M2 + M3 + M6
        
        # Re-doing W vectors carefully.
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1, C11 -= 1
        # M3: C01 += 1, C11 += 1
        # M4: C00 += 1, C10 += 1
        # M5: C00 -= 1, C01 += 1
        # M6: C11 += 1
        # M7: C00 += 1
        
        # Let's just define the 7 U, V vectors and the resulting W contributions.
        # It's safer to generate them programmatically or hardcode the full 49.
        # Given the complexity, I will hardcode the 7 base terms' U and V, and the W coefficients.
        
        base_u = [
            [1, 0, 0, 1], # M1 A
            [0, 0, 1, 1], # M2 A
            [1, 0, 0, 0], # M3 A
            [0, 0, 0, 1], # M4 A
            [1, 1, 0, 0], # M5 A
            [0, 0, 1, -1],# M6 A
            [0, 1, 0, -1] # M7 A
        ]
        base_v = [
            [1, 0, 0, 1], # M1 B
            [1, 0, 0, 0], # M2 B
            [0, 1, 0, -1],# M3 B
            [0, 0, 1, -1],# M4 B
            [0, 0, 0, 1], # M5 B
            [1, 1, 0, 0], # M6 B
            [0, 0, 1, 1]  # M7 B
        ]
        # W coefficients for C = [C00, C01, C10, C11]
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        base_w = [
            [1, 0, 0, 1], # M1: C00+1, C11+1
            [0, 0, 1, -1],# M2: C10+1, C11-1
            [0, 1, 0, 1], # M3: C01+1, C11+1
            [1, 0, 1, 0], # M4: C00+1, C10+1
            [-1, 1, 0, 0],# M5: C00-1, C01+1
            [0, 0, 0, 1], # M6: C11+1
            [1, 0, 0, 0]  # M7: C00+1
        ]
        return base_u, base_v, base_w

    base_u, base_v, base_w = get_base_strassen()
    
    # Outer Strassen coefficients for 4x4x4
    # Blocks: 00, 01, 10, 11
    # M1: (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2: (A10+A11)B00      -> C10+=M2, C11-=M2
    # M3: A00(B01-B11)      -> C01+=M3, C11+=M3
    # M4: A11(B10-B00)      -> C00+=M4, C10+=M4
    # M5: (A00+A01)B11      -> C00-=M5, C01+=M5
    # M6: (A10-A00)(B00+B01)-> C11+=M6
    # M7: (A01-A11)(B10+B11)-> C00+=M7
    
    outer_u_blocks = [
        [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], # M1: A00+A11 (block 00 has 1, block 11 has 1)
        # Actually, let's represent outer_u as a 4x4 matrix of coefficients for the 4 blocks.
        # Each block is 2x2.
        # It's easier to iterate.
    ]
    
    # Let's define the outer coefficients as a list of 7, each containing 4 vectors (for the 4 blocks)
    # Each vector is length 4 (flattened 2x2 block).
    # For M1: A00 + A11.
    # Block 00 coeff: [1,0,0,1]? No, the block itself is added.
    # The "linear form" L_A1 = A00 + A11.
    # So the coefficient for elements in A00 is 1, A11 is 1, others 0.
    
    outer_u = [
        # M1: A00 + A11
        [[1,0,0,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]], # This is wrong. A00 is a 2x2 matrix.
        # The coefficient for the *entire block* A00 is 1.
        # So for block 00, the 2x2 coefficient matrix is Identity? No, it's 1 * Identity?
        # No, L_A1 = A00 + A11.
        # Element (0,0) of L_A1 is A00[0,0] + A11[0,0].
        # So the coefficient for A00[0,0] is 1.
        # So the 2x2 coefficient matrix for block 00 is [[1,0],[0,1]].
        # For block 11 is [[1,0],[0,1]].
    ]
    
    # Correct representation:
    # outer_u[s] is a list of 4 vectors (length 4) representing the coefficient matrices for blocks 00, 01, 10, 11.
    # Each vector is flattened 2x2.
    
    def make_block_coeff(val):
        if val == 1:
            return [1, 0, 0, 1]
        elif val == -1:
            return [-1, 0, 0, -1]
        else:
            return [0, 0, 0, 0]
            
    # M1: A00 + A11
    ou_0 = [make_block_coeff(1), make_block_coeff(0), make_block_coeff(0), make_block_coeff(1)]
    # M2: A10 + A11
    ou_1 = [make_block_coeff(0), make_block_coeff(0), make_block_coeff(1), make_block_coeff(1)]
    # M3: A00
    ou_2 = [make_block_coeff(1), make_block_coeff(0), make_block_coeff(0), make_block_coeff(0)]
    # M4: A11
    ou_3 = [make_block_coeff(0), make_block_coeff(0), make_block_coeff(0), make_block_coeff(1)]
    # M5: A00 + A01
    ou_4 = [make_block_coeff(1), make_block_coeff(1), make_block_coeff(0), make_block_coeff(0)]
    # M6: A10 - A00
    ou_5 = [make_block_coeff(-1), make_block_coeff(0), make_block_coeff(1), make_block_coeff(0)]
    # M7: A01 - A11
    ou_6 = [make_block_coeff(0), make_block_coeff(1), make_block_coeff(0), make_block_coeff(-1)]
    
    outer_u = [ou_0, ou_1, ou_2, ou_3, ou_4, ou_5, ou_6]
    
    # Outer V
    # M1: B00 + B11
    ov_0 = [make_block_coeff(1), make_block_coeff(0), make_block_coeff(0), make_block_coeff(1)]
    # M2: B00
    ov_1 = [make_block_coeff(1), make_block_coeff(0), make_block_coeff(0), make_block_coeff(0)]
    # M3: B01 - B11
    ov_2 = [make_block_coeff(0), make_block_coeff(1), make_block_coeff(0), make_block_coeff(-1)]
    # M4: B10 - B00
    ov_3 = [make_block_coeff(-1), make_block_coeff(0), make_block_coeff(1), make_block_coeff(0)]
    # M5: B11
    ov_4 = [make_block_coeff(0), make_block_coeff(0), make_block_coeff(0), make_block_coeff(1)]
    # M6: B00 + B01
    ov_5 = [make_block_coeff(1), make_block_coeff(1), make_block_coeff(0), make_block_coeff(0)]
    # M7: B10 + B11
    ov_6 = [make_block_coeff(0), make_block_coeff(0), make_block_coeff(1), make_block_coeff(1)]
    
    outer_v = [ov_0, ov_1, ov_2, ov_3, ov_4, ov_5, ov_6]
    
    # Outer W (Reconstruction coefficients for C blocks)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # w_block[s][block_idx] is the coefficient for P_s in that block.
    # P_s is a 2x2 matrix. The coefficient applies to the whole 2x2 matrix.
    # So the 2x2 coefficient matrix is coeff * Identity.
    
    def make_w_block_coeff(val):
        if val == 1:
            return [1, 0, 0, 1]
        elif val == -1:
            return [-1, 0, 0, -1]
        else:
            return [0, 0, 0, 0]
            
    # outer_w[s] is a list of 4 vectors (length 4) for blocks 00, 01, 10, 11.
    ow_0 = [make_w_block_coeff(1), make_w_block_coeff(0), make_w_block_coeff(0), make_w_block_coeff(1)] # M1: C00, C11
    ow_1 = [make_w_block_coeff(0), make_w_block_coeff(0), make_w_block_coeff(1), make_w_block_coeff(-1)] # M2: C10, C11
    ow_2 = [make_w_block_coeff(0), make_w_block_coeff(1), make_w_block_coeff(0), make_w_block_coeff(1)] # M3: C01, C11
    ow_3 = [make_w_block_coeff(1), make_w_block_coeff(0), make_w_block_coeff(1), make_w_block_coeff(0)] # M4: C00, C10
    ow_4 = [make_w_block_coeff(-1), make_w_block_coeff(1), make_w_block_coeff(0), make_w_block_coeff(0)] # M5: C00, C01
    ow_5 = [make_w_block_coeff(0), make_w_block_coeff(0), make_w_block_coeff(0), make_w_block_coeff(1)] # M6: C11
    ow_6 = [make_w_block_coeff(1), make_w_block_coeff(0), make_w_block_coeff(0), make_w_block_coeff(0)] # M7: C00
    
    outer_w = [ow_0, ow_1, ow_2, ow_3, ow_4, ow_5, ow_6]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Combine outer and base coefficients
            # U: outer_u[s][block] * base_u[t]
            # V: outer_v[s][block] * base_v[t]
            # W: outer_w[s][block] * base_w[t]
            
            for b in range(4): # Block index 0..3
                # Global indices for U
                # Block b corresponds to rows 2*(b//2) and 2*(b//2)+1?
                # Blocks: 00 (0), 01 (1), 10 (2), 11 (3)
                # Row block r = b // 2, Col block c = b % 2
                # Global row offset: 2 * (b // 2)
                # Global col offset: 2 * (b % 2)
                
                u_block = outer_u[s][b]
                v_block = outer_v[s][b]
                w_block = outer_w[s][b]
                
                # u_block is length 4 (2x2 flattened)
                # base_u[t] is length 4
                # The product is element-wise? No.
                # The "linear form" L_A is a matrix.
                # The base Strassen term uses a specific linear combination of elements of L_A.
                # So the coefficient for a specific element A[i,j] is:
                # (Coefficient of A[i,j] in L_A) * (Coefficient of L_A[i',j'] in base term)
                # Wait. L_A is a 2x2 matrix.
                # base_u[t] is a vector of coefficients for the 4 elements of the 2x2 input matrix to the base Strassen.
                # So, for a specific global element A[global_i, global_j]:
                # It belongs to block b.
                # Its local index in the block is li, lj.
                # The coefficient in L_A is u_block[li*2 + lj].
                # The coefficient in the base term is base_u[t][li*2 + lj].
                # So the total coefficient is u_block[li*2 + lj] * base_u[t][li*2 + lj].
                
                # We iterate over the 4 local elements
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        coeff_u = u_block[local_idx] * base_u[t][local_idx]
                        coeff_v = v_block[local_idx] * base_v[t][local_idx]
                        
                        if coeff_u != 0:
                            global_i = 2 * (b // 2) + li
                            global_j = 2 * (b % 2) + lj
                            U[global_i * m + global_j][col] += coeff_u
                        
                        if coeff_v != 0:
                            global_j_v = 2 * (b % 2) + lj # B col offset
                            global_i_v = 2 * (b // 2) + li # B row offset? 
                            # B is 4x4. Blocks are same layout.
                            # B block b corresponds to rows 2*(b//2).. and cols 2*(b%2)..
                            # So global row for B is 2*(b//2)+li, global col is 2*(b%2)+lj.
                            # But B index is j*p + k. Here m=4, p=4 (for this part).
                            # So index is (2*(b//2)+li) * 4 + (2*(b%2)+lj).
                            V[(2*(b//2)+li) * p + (2*(b%2)+lj)][col] += coeff_v
                        
                        # W
                        # The base term t contributes to local C elements.
                        # base_w[t] is length 4.
                        # The outer term s contributes to block b with coefficient w_block.
                        # w_block is a 2x2 matrix of coefficients.
                        # So for local C element (li, lj), the coefficient is w_block[local_idx] * base_w[t][local_idx].
                        coeff_w = w_block[local_idx] * base_w[t][local_idx]
                        
                        if coeff_w != 0:
                            # Global C index
                            # Block b is C block.
                            # Global row: 2*(b//2) + li
                            # Global col: 2*(b%2) + lj
                            # W index: k*n + i = global_col * 4 + global_row
                            global_c_row = 2 * (b // 2) + li
                            global_c_col = 2 * (b % 2) + lj
                            W[global_c_col * n + global_c_row][col] += coeff_w
            col += 1
            
    # --- Part 2: Schoolbook for 4x4x1 (5th column) ---
    # C[:, 4] = A * B[:, 4]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
