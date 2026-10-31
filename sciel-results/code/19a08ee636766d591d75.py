import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 blocks
    # Outer Strassen
    # M1 = (A11 - A00)(B00 + B11)
    # M2 = (A00 + A10)B00
    # M3 = A11(B10 - B00)
    # M4 = A00(B01 - B11)
    # M5 = (A01 + A11)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Inner Strassen (same structure)
    
    # Helper to add to U, V, W
    def add_tensor(col_idx, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: dict {global_a_index: coeff}
        # b_coeffs: dict {global_b_index: coeff}
        # c_coeffs: dict {global_c_index: coeff}
        for idx, val in a_coeffs.items():
            U[idx][col_idx] += val
        for idx, val in b_coeffs.items():
            V[idx][col_idx] += val
        for idx, val in c_coeffs.items():
            W[idx][col_idx] += val

    # Pre-compute global indices for blocks
    # Block (R, C) contains indices (2R+dr, 2C+dc) for dr,dc in 0,1
    # Global index for A: r*4 + c
    # Global index for B: r*4 + c (row r of B, col c) -- wait, B is m x p = 4x4.
    # Global index for C: r*4 + c.
    
    # Map block (R, C) to list of 4 global indices
    def block_indices(R, C, matrix_type):
        # matrix_type 'A', 'B', 'C'
        # A: row r, col c. r in 0..3, c in 0..3
        # B: row r, col c.
        # C: row r, col c.
        # Note: In tensor decomposition, U is indexed by A's rows?
        # U shape (n*m, rank) -> U[i*m+j] corresponds to A[i,j].
        # V shape (m*p, rank) -> V[j*p+k] corresponds to B[j,k].
        # W shape (p*n, rank) -> W[k*n+i] corresponds to C[i,k].
        # Task says: W[k*n+i, r]. This is row index k*n+i.
        # So W row index is k*n + i.
        # C[i,k] corresponds to W[k*n + i].
        
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = 2*R + dr
                c = 2*C + dc
                if matrix_type == 'A':
                    # U index: i*m + j = r*4 + c
                    indices.append(r * 4 + c)
                elif matrix_type == 'B':
                    # V index: j*p + k = r*4 + c
                    indices.append(r * 4 + c)
                elif matrix_type == 'C':
                    # W index: k*n + i = c*4 + r  (since k=c, i=r)
                    indices.append(c * 4 + r)
        return indices

    # Outer products definition
    # Each is (U_blocks, V_blocks, C_block_coeffs)
    # U_blocks: dict {(R,C): coeff}
    # V_blocks: dict {(R,C): coeff}
    # C_block_coeffs: dict {(R,C): coeff} for the result C block
    
    outer_products = [
        # M1
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}), # M1 contributes to C00 and C11?
        # Wait, C00 = M1 + M4 - M5 + M7. C11 = M1 - M2 + M4 + M6.
        # So M1 contributes to C00 with 1 and C11 with 1.
        # M2 contributes to C01 with 1 and C11 with -1.
        # M3 contributes to C01 with 1.
        # M4 contributes to C00 with 1 and C11 with 1.
        # M5 contributes to C10 with 1 and C00 with -1.
        # M6 contributes to C10 with 1 and C11 with 1.
        # M7 contributes to C00 with 1.
        
        # Let's re-verify Strassen C formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3
        # C10 = M6 + M5
        # C11 = M1 - M2 + M4 + M6
        
        # So:
        # M1: C00:1, C11:1
        # M2: C01:1, C11:-1
        # M3: C01:1
        # M4: C00:1, C11:1
        # M5: C10:1, C00:-1
        # M6: C10:1, C11:1
        # M7: C00:1
    ]
    
    outer_products = [
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(0,0): 1, (1,0): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}), # M2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}), # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,1): 1}), # M4
        ({(0,1): 1, (1,1): 1}, {(1,1): 1}, {(1,0): 1, (0,0): -1}), # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}), # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}) # M7
    ]
    
    # Inner Strassen products definition
    # For a 2x2 matrix U with entries u00, u01, u10, u11
    # and V with v00, v01, v10, v11
    # P1 = (u11 - u00)(v00 + v11) -> contributes to w00:1, w11:1
    # P2 = (u00 + u10)v00 -> contributes to w01:1, w11:-1
    # P3 = u11(v10 - v00) -> contributes to w01:1
    # P4 = u00(v01 - v11) -> contributes to w00:1, w11:1
    # P5 = (u01 + u11)v11 -> contributes to w10:1, w00:-1
    # P6 = (u10 - u00)(v00 + v01) -> contributes to w10:1, w11:1
    # P7 = (u01 - u11)(v10 + v11) -> contributes to w00:1
    
    inner_products = [
        # (U_entry_coeffs, V_entry_coeffs, W_entry_coeffs)
        # U_entry_coeffs: dict {(r,c): coeff}
        # W_entry_coeffs: dict {(r,c): coeff}
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # P1
        ({(0,0): 1, (1,0): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}), # P2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}), # P3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,1): 1}), # P4
        ({(0,1): 1, (1,1): 1}, {(1,1): 1}, {(1,0): 1, (0,0): -1}), # P5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}), # P6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}) # P7
    ]
    
    col = 0
    for i in range(7):
        u_blocks, v_blocks, c_blocks = outer_products[i]
        for k in range(7):
            u_entries, v_entries, w_entries = inner_products[k]
            
            # Build L_A
            # L_A is linear combo of global A indices
            a_coeffs = {}
            for (u_r, u_c), u_coeff in u_entries.items():
                # u_r, u_c are local coords in the 2x2 U matrix
                # U[u_r, u_c] is a linear combo of A blocks
                # Wait, U is the result of Outer product's A part.
                # U[u_r, u_c] = sum_{block} alpha_block A_block
                # No, U is a 2x2 matrix. Each entry U[r,c] is a scalar?
                # No, in Strassen, U and V are matrices.
                # Here, for the INNER Strassen, U and V are 2x2 matrices of SCALARS.
                # But these scalars are linear combos of A elements?
                # No.
                # Let's trace back.
                # Outer M_i = (Sum A_blocks) (Sum B_blocks).
                # Let X = Sum A_blocks, Y = Sum B_blocks.
                # X and Y are 2x2 matrices of scalars (linear combos of A/B elements).
                # We apply Inner Strassen to X and Y.
                # Inner Strassen computes X Y.
                # The "U" in Inner Strassen is X.
                # So X[r,c] is a scalar linear combo of A elements.
                # Specifically, X[r,c] = sum_{block} alpha_block A_block.
                # But A_block is a 2x2 matrix of scalars.
                # So X is a 2x2 matrix of 2x2 matrices?
                # No.
                # The Outer Strassen operates on 2x2 BLOCKS.
                # The "entries" of the block matrix are 2x2 matrices.
                # So X is a 2x2 matrix of 2x2 matrices.
                # Inner Strassen operates on the 2x2 matrix of 2x2 matrices?
                # No, Inner Strassen is for 2x2 scalar matrices.
                # We are decomposing the product of two 2x2 matrices of scalars (which are linear combos of A/B elements).
                # So X[r,c] is a scalar.
                # Wait.
                # A is 4x4.
                # A_block(R,C) is 2x2.
                # M_i = (A_block(1,1) - A_block(0,0)) (B_block(0,0) + B_block(1,1)).
                # Let X = A_block(1,1) - A_block(0,0). X is 2x2.
                # Let Y = B_block(0,0) + B_block(1,1). Y is 2x2.
                # M_i = X Y.
                # We decompose X Y using Strassen for 2x2.
                # Strassen for 2x2 uses 7 products of SCALARS.
                # But X and Y are 2x2 matrices.
                # The Strassen formula uses entries of X and Y.
                # X[0,0] is a scalar (linear combo of A elements).
                # So the 7 products are products of scalars.
                # Each product is a rank-1 tensor.
                # So we need 7 rank-1 tensors for each M_i.
                # Total 49.
                
                # So, for Inner Strassen product k:
                # It is a product of (linear combo of X entries) and (linear combo of Y entries).
                # X entry is a linear combo of A elements.
                # Y entry is a linear combo of B elements.
                # So we need to map X entries to A elements.
                
                # How to get X entries?
                # X = sum_{block} alpha_block A_block.
                # X[r,c] = sum_{block} alpha_block (A_block)[r,c].
                # (A_block)[r,c] is a specific global A element.
                # Global index for A_block(R,C)[r,c] is (2R+r)*4 + (2C+c).
                
                # So for a given u_r, u_c (local in X):
                # X[u_r, u_c] = sum_{(R,C), alpha in u_blocks} alpha * A_global(2R+u_r, 2C+u_c).
                
                # So for Inner product k, we have u_entries: {(u_r, u_c): coeff}.
                # The linear combo of X is sum_{(u_r, u_c)} coeff * X[u_r, u_c].
                # = sum_{(u_r, u_c)} coeff * sum_{(R,C), alpha in u_blocks} alpha * A_global(2R+u_r, 2C+u_c).
                # = sum_{global_a} (sum_{...} coeff * alpha) A_global.
                
                pass
            
            # Logic to build a_coeffs
            a_coeffs = {}
            for (u_r, u_c), u_coeff in u_entries.items():
                for (R, C), alpha in u_blocks.items():
                    global_r = 2 * R + u_r
                    global_c = 2 * C + u_c
                    global_idx = global_r * 4 + global_c
                    val = u_coeff * alpha
                    a_coeffs[global_idx] = a_coeffs.get(global_idx, 0) + val

            b_coeffs = {}
            for (v_r, v_c), v_coeff in v_entries.items():
                for (R, C), beta in v_blocks.items():
                    global_r = 2 * R + v_r
                    global_c = 2 * C + v_c
                    global_idx = global_r * 4 + global_c
                    val = v_coeff * beta
                    b_coeffs[global_idx] = b_coeffs.get(global_idx, 0) + val

            c_coeffs = {}
            for (w_r, w_c), w_coeff in w_entries.items():
                for (R, C), gamma in c_blocks.items():
                    # w_r, w_c are local in the 2x2 result block
                    # The result block is at (R,C) in the outer 2x2 grid.
                    # So global C element is at (2R+w_r, 2C+w_c).
                    global_r = 2 * R + w_r
                    global_c = 2 * C + w_c
                    # W index: k*n + i = global_c * 4 + global_r
                    global_idx = global_c * 4 + global_r
                    val = w_coeff * gamma
                    c_coeffs[global_idx] = c_coeffs.get(global_idx, 0) + val

            add_tensor(col, a_coeffs, b_coeffs, c_coeffs)
            col += 1
            
    return {"U": U, "V": V, "W": W}
