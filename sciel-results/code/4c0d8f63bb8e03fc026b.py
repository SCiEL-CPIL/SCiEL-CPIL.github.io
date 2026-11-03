"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    
    # We decompose into two 4x4x4 multiplications.
    # Each 4x4x4 has rank 49.
    # Total rank = 98.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen coefficients for 2x2x2
    # X, Y are 2x2 matrices. Z = XY.
    # S_1 = (X00+X11)(Y00+Y11)
    # S_2 = (X10+X11)Y00
    # S_3 = X00(Y01-Y11)
    # S_4 = X11(Y10-Y00)
    # S_5 = (X00+X10)Y11
    # S_6 = (X01-X11)(Y10+Y11)
    # S_7 = (X00-X10)(Y00+Y01)
    
    # coeff_X[k][r][c] is coefficient of X[r][c] in S_k
    coeff_X = [
        { (0,0): 1, (1,1): 1 }, # S1
        { (1,0): 1, (1,1): 1 }, # S2
        { (0,0): 1 },           # S3
        { (1,1): 1 },           # S4
        { (0,0): 1, (1,0): 1 }, # S5
        { (0,1): 1, (1,1): -1 },# S6
        { (0,0): 1, (1,0): -1 } # S7
    ]
    
    coeff_Y = [
        { (0,0): 1, (1,1): 1 }, # S1
        { (0,0): 1 },           # S2
        { (0,1): 1, (1,1): -1 },# S3
        { (1,0): 1, (0,0): -1 },# S4
        { (1,1): 1 },           # S5
        { (1,0): 1, (1,1): 1 }, # S6
        { (0,0): 1, (0,1): 1 }  # S7
    ]
    
    # Z entries are linear combos of S_k
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    coeff_Z = [
        { 0: 1, 3: 1, 4: -1, 6: 1 }, # Z00 from S0..S6 (indices 0-6)
        { 2: 1, 4: 1 },               # Z01
        { 1: 1, 3: 1 },               # Z10
        { 0: 1, 1: -1, 2: 1, 5: 1 }   # Z11
    ]
    
    # Outer Strassen for 4x4x4
    # A, B are 4x4 matrices, split into 2x2 blocks of 2x2 matrices.
    # M_1 = (A00+A11)(B00+B11)
    # M_2 = (A10+A11)B00
    # M_3 = A00(B01-B11)
    # M_4 = A11(B10-B00)
    # M_5 = (A00+A10)B11
    # M_6 = (A01-A11)(B10+B11)
    # M_7 = (A00-A10)(B00+B01)
    
    # A_s is a 2x2 matrix of 2x2 blocks.
    # A_s[r][c] is a 2x2 matrix formed from A's blocks.
    # A's blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    
    # Define A_s structure for s=0..6
    # Each entry is a dict mapping (block_r, block_c) to coefficient
    # block indices: 00, 01, 10, 11
    outer_A_blocks = [
        { (0,0): {(0,0):1, (1,1):1}, (0,1): {(0,0):1, (1,1):1}, (1,0): {(0,0):1, (1,1):1}, (1,1): {(0,0):1, (1,1):1} }, # M1: (A00+A11)
        { (0,0): {(1,0):1, (1,1):1}, (0,1): {(1,0):1, (1,1):1}, (1,0): {(1,0):1, (1,1):1}, (1,1): {(1,0):1, (1,1):1} }, # M2: (A10+A11)
        { (0,0): {(0,0):1}, (0,1): {(0,0):1}, (1,0): {(0,0):1}, (1,1): {(0,0):1} }, # M3: A00
        { (0,0): {(1,1):1}, (0,1): {(1,1):1}, (1,0): {(1,1):1}, (1,1): {(1,1):1} }, # M4: A11
        { (0,0): {(0,0):1, (1,0):1}, (0,1): {(0,0):1, (1,0):1}, (1,0): {(0,0):1, (1,0):1}, (1,1): {(0,0):1, (1,0):1} }, # M5: (A00+A10)
        { (0,0): {(0,1):1, (1,1):-1}, (0,1): {(0,1):1, (1,1):-1}, (1,0): {(0,1):1, (1,1):-1}, (1,1): {(0,1):1, (1,1):-1} }, # M6: (A01-A11)
        { (0,0): {(0,0):1, (1,0):-1}, (0,1): {(0,0):1, (1,0):-1}, (1,0): {(0,0):1, (1,0):-1}, (1,1): {(0,0):1, (1,0):-1} }  # M7: (A00-A10)
    ]
    
    outer_B_blocks = [
        { (0,0): {(0,0):1, (1,1):1}, (0,1): {(0,0):1, (1,1):1}, (1,0): {(0,0):1, (1,1):1}, (1,1): {(0,0):1, (1,1):1} }, # M1: (B00+B11)
        { (0,0): {(0,0):1}, (0,1): {(0,0):1}, (1,0): {(0,0):1}, (1,1): {(0,0):1} }, # M2: B00
        { (0,0): {(0,1):1, (1,1):-1}, (0,1): {(0,1):1, (1,1):-1}, (1,0): {(0,1):1, (1,1):-1}, (1,1): {(0,1):1, (1,1):-1} }, # M3: (B01-B11)
        { (0,0): {(1,0):1, (0,0):-1}, (0,1): {(1,0):1, (0,0):-1}, (1,0): {(1,0):1, (0,0):-1}, (1,1): {(1,0):1, (0,0):-1} }, # M4: (B10-B00)
        { (0,0): {(1,1):1}, (0,1): {(1,1):1}, (1,0): {(1,1):1}, (1,1): {(1,1):1} }, # M5: B11
        { (0,0): {(1,0):1, (1,1):1}, (0,1): {(1,0):1, (1,1):1}, (1,0): {(1,0):1, (1,1):1}, (1,1): {(1,0):1, (1,1):1} }, # M6: (B10+B11)
        { (0,0): {(0,0):1, (0,1):1}, (0,1): {(0,0):1, (0,1):1}, (1,0): {(0,0):1, (0,1):1}, (1,1): {(0,0):1, (0,1):1} }  # M7: (B00+B01)
    ]
    
    # C assembly
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # coeff_C[block_i][block_j][s]
    coeff_C = [
        { 0: {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}, 1: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 2: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 3: {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}, 4: {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 0}, 5: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 6: {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0} }, # C00
        { 0: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 1: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 2: {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}, 3: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 4: {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}, 5: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 6: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0} }, # C01
        { 0: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 1: {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}, 2: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 3: {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}, 4: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 5: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 6: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0} }, # C10
        { 0: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}, 1: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): -1}, 2: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}, 3: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 4: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}, 5: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}, 6: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0} }  # C11
    ]
    
    def build_one_instance(offset_U, offset_V, offset_W, b_start, c_start):
        # b_start: 0 or 4 (column offset in B and C)
        # offset_U: 0 or 49
        # offset_V: 0 or 49
        # offset_W: 0 or 49
        
        for s in range(7):
            for k in range(7):
                col = offset_U + s * 7 + k
                
                # U vector
                # U[i, j] where i,j in 0..3
                # A_s[r][c] is a 2x2 matrix.
                # X_{r,c} = A_s[r][c]
                # S_k = sum_{r,c} coeff_X[k][r,c] * X_{r,c}
                # X_{r,c} is a 2x2 matrix.
                # A_s[r][c] is formed from A's blocks.
                
                for r in range(2):
                    for c in range(2):
                        # Get coefficients from inner decomposition for X_{r,c}
                        if (r, c) in coeff_X[k]:
                            x_coeff = coeff_X[k][(r, c)]
                        else:
                            continue
                        
                        # A_s[r][c] is a 2x2 matrix.
                        # Its entries are linear combos of A's blocks.
                        # A_s structure for s is in outer_A_blocks[s]
                        # outer_A_blocks[s][(r,c)] is a dict { (block_r, block_c): coeff }
                        block_coeffs = outer_A_blocks[s][(r, c)]
                        
                        for (br, bc), a_coeff in block_coeffs.items():
                            # A's block (br, bc) is a 2x2 matrix.
                            # We need to add a_coeff * x_coeff to U for the entries of this block.
                            # Block (br, bc) corresponds to A rows br*2..br*2+1, cols bc*2..bc*2+1
                            for dr in range(2):
                                for dc in range(2):
                                    i = br * 2 + dr
                                    j = bc * 2 + dc
                                    U[i * m + j][col] += a_coeff * x_coeff
                
                # V vector
                # V[j, k] where j in 0..3 (rows of B), k in 0..7 (cols of B)
                # We are handling a 4x4x4 instance.
                # B is 4x4 (cols 0-3 relative to instance).
                # Global col in B is b_start + local_col.
                
                for r in range(2):
                    for c in range(2):
                        if (r, c) in coeff_Y[k]:
                            y_coeff = coeff_Y[k][(r, c)]
                        else:
                            continue
                        
                        block_coeffs = outer_B_blocks[s][(r, c)]
                        
                        for (br, bc), b_coeff in block_coeffs.items():
                            for dr in range(2):
                                for dc in range(2):
                                    j = br * 2 + dr
                                    local_k = bc * 2 + dc
                                    global_k = b_start + local_k
                                    V[j * p + global_k][col] += b_coeff * y_coeff
                
                # W vector
                # W[k, i] where k in 0..7 (rows of C), i in 0..3 (cols of C)
                # C is 4x4 (cols 0-3 relative to instance).
                # Global col in C is c_start + local_col.
                # Note: Task says W[k*n+i, r]. k is row of C (0..7), i is col of C (0..3).
                # Wait, C is 4x8. Rows 0..3. Cols 0..7.
                # W index: k*n + i. k is row (0..3), i is col (0..7).
                
                # M_s is a 2x2 matrix of 2x2 matrices? No.
                # M_s is a 2x2 matrix.
                # Each entry M_{s, r, c} is a 2x2 matrix?
                # No, in the outer Strassen, M_s is the result of multiplying two 2x2 matrices of 2x2 blocks.
                # So M_s is a 2x2 matrix of 2x2 matrices.
                # So M_s has 4 entries, each a 2x2 matrix.
                
                # Inner decomposition gives S_k.
                # M_s entries are linear combos of S_k.
                # M_{s, r, c} (which is a 2x2 matrix) is formed by:
                # M_{s, r, c}[dr, dc] = sum_{inner_k} coeff_Z[inner_k][dr, dc] * S_inner?
                # No, coeff_Z is for the 2x2 matrix result Z of the inner multiplication.
                # The inner multiplication produces a 2x2 matrix Z.
                # Z[dr, dc] = sum_{k} coeff_Z[dr, dc][k] * S_k.
                
                # So M_{s, r, c} is a 2x2 matrix.
                # Let's call its entries M_{s, r, c, dr, dc}.
                # M_{s, r, c, dr, dc} = sum_{k=0}^6 coeff_Z[dr, dc][k] * S_k.
                
                # Wait, coeff_Z is indexed by [result_row, result_col][scalar_index].
                # coeff_Z[dr][dc] is a dict {k: coeff}.
                
                # Now, how does M_s contribute to C?
                # C_block(i, j) = sum_s coeff_C[i][j][s] * M_s.
                # C_block(i, j) is a 2x2 matrix.
                # C_block(i, j)[dr, dc] = sum_s coeff_C[i][j][s] * M_{s, dr, dc}.
                
                # So for a specific (s, k):
                # The contribution to C_block(i, j)[dr, dc] is:
                # coeff_C[i][j][s] * (coeff_Z[dr][dc][k])
                
                # And this contributes to global C[ i*2+dr, j*2+dc ].
                # Global row in C: i*2+dr.
                # Global col in C: c_start + (j*2+dc).
                
                for i_block in range(2):
                    for j_block in range(2):
                        c_coeff_s = coeff_C[i_block][j_block].get(s, 0)
                        if c_coeff_s == 0: continue
                        
                        for dr in range(2):
                            for dc in range(2):
                                z_coeff_k = coeff_Z[dr][dc].get(k, 0)
                                if z_coeff_k == 0: continue
                                
                                w_val = c_coeff_s * z_coeff_k
                                if w_val != 0:
                                    global_i = i_block * 2 + dr
                                    global_j = c_start + j_block * 2 + dc
                                    W[global_i * n + global_j][col] += w_val

    # First instance: B cols 0-3, C cols 0-3
    build_one_instance(0, 0, 0, 0, 0)
    # Second instance: B cols 4-7, C cols 4-7
    build_one_instance(49, 49, 49, 4, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
