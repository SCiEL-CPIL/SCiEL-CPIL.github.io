"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm for each 2x2 block multiplication.
    A (2x4) is split into two 2x2 blocks: A = [A1 | A2]
    B (4x6) is split into two 2x6 blocks: B = [B1; B2]
    Then A1*B1 is a 2x2 * 2x6 = 2x6 product.
    A2*B2 is a 2x2 * 2x6 = 2x6 product.
    
    For each 2x2 * 2x6 product, we split the 2x6 into three 2x2 blocks.
    Each 2x2 * 2x2 uses Strassen (rank 7).
    
    Total: 2 (A blocks) * 3 (B sub-blocks per A block) * 7 (Strassen rank) = 42.
    
    Output convention: W[k*n + i, r]
    U shape: (n*m, rank) = (8, 42)
    V shape: (m*p, rank) = (24, 42)
    W shape: (p*n, rank) = (12, 42)
    """
    n, m, p = 2, 4, 6
    
    # Strassen's 7 factors for 2x2 matrix multiplication:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M3 - M5 + M7
    #
    # Where:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each Strassen factor, we define the linear combinations:
    # (a_indices, a_coeffs, b_indices, b_coeffs)
    # where the factor = sum(a_coeffs[i] * A[a_indices[i]]) * sum(b_coeffs[j] * B[b_indices[j]])
    
    strassen_factors = [
        # M1 = (A11 + A22)(B11 + B22)
        {
            'a_indices': [(0, 0), (1, 1)],
            'a_coeffs': [1, 1],
            'b_indices': [(0, 0), (1, 1)],
            'b_coeffs': [1, 1],
            'c_coeff': 1,   # contributes +1 to C11 and C22
            'c_position': 'M1'
        },
        # M2 = (A21 + A22)B11
        {
            'a_indices': [(1, 0), (1, 1)],
            'a_coeffs': [1, 1],
            'b_indices': [(0, 0)],
            'b_coeffs': [1],
            'c_coeff': 1,   # contributes +1 to C12
            'c_position': 'M2'
        },
        # M3 = A11(B21 - B22)
        {
            'a_indices': [(0, 0)],
            'a_coeffs': [1],
            'b_indices': [(1, 0), (1, 1)],
            'b_coeffs': [1, -1],
            'c_coeff': 1,   # contributes +1 to C21
            'c_position': 'M3'
        },
        # M4 = A22(B12 - B11)
        {
            'a_indices': [(1, 1)],
            'a_coeffs': [1],
            'b_indices': [(0, 1), (0, 0)],
            'b_coeffs': [1, -1],
            'c_coeff': -1,  # contributes -1 to C11, +1 to C12
            'c_position': 'M4'
        },
        # M5 = (A11 + A12)B22
        {
            'a_indices': [(0, 0), (0, 1)],
            'a_coeffs': [1, 1],
            'b_indices': [(1, 1)],
            'b_coeffs': [1],
            'c_coeff': 1,   # contributes +1 to C21, -1 to C22
            'c_position': 'M5'
        },
        # M6 = (A21 - A11)(B11 + B12)
        {
            'a_indices': [(1, 0), (0, 0)],
            'a_coeffs': [1, -1],
            'b_indices': [(0, 0), (0, 1)],
            'b_coeffs': [1, 1],
            'c_coeff': 1,   # contributes +1 to C11
            'c_position': 'M6'
        },
        # M7 = (A12 - A22)(B21 + B22)
        {
            'a_indices': [(0, 1), (1, 1)],
            'a_coeffs': [1, -1],
            'b_indices': [(1, 0), (1, 1)],
            'b_coeffs': [1, 1],
            'c_coeff': 1,   # contributes +1 to C11, -1 to C22
            'c_position': 'M7'
        },
    ]
    
    # How each Strassen factor contributes to C entries:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M3 - M5 + M7
    
    # For each block multiplication, we need to track which C position each factor goes to.
    # We'll generate terms for each (A_block, B_subblock) pair.
    
    # A is 2x4, split into A1 (cols 0-1) and A2 (cols 2-3)
    # B is 4x6, split into B1 (rows 0-1) and B2 (rows 2-3)
    # B1 is 2x6, split into B11 (cols 0-1), B12 (cols 2-3), B13 (cols 4-5)
    # B2 is 2x6, split into B21 (cols 0-1), B22 (cols 2-3), B23 (cols 4-5)
    
    # Block definitions: (a_col_offset, b_row_offset, b_col_offset)
    # a_col_offset: which 2x2 block of A (0 for A1, 1 for A2)
    # b_row_offset: which 2x2 block of B rows (0 for B1, 1 for B2)
    # b_col_offset: which 2x2 block of B cols (0, 1, 2 for cols 0-1, 2-3, 4-5)
    
    blocks = []
    for a_block in range(2):       # A1, A2
        for b_row_block in range(2):  # B1, B2
            for b_col_block in range(3):  # 3 sub-blocks in each 2x6
                blocks.append((a_block, b_row_block, b_col_block))
    
    # Each block gives 7 Strassen terms, so total rank = 6 * 7 = 42
    
    rank = len(blocks) * 7
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    col = 0
    for (a_block, b_row_block, b_col_block) in blocks:
        a_col_off = a_block * 2      # 0 or 2
        b_row_off = b_row_block * 2  # 0 or 2
        b_col_off = b_col_block * 2  # 0, 2, or 4
        
        for sf in strassen_factors:
            # This Strassen factor corresponds to a scalar multiplication
            # We add terms to U, V, W for this factor
            
            # For U: A entries used
            # A global index: i * m + j, where i is row (0 or 1), j is col
            # Local A indices (li, lj) map to global (i, a_col_off + lj)
            for (li, lj), coeff in zip(sf['a_indices'], sf['a_coeffs']):
                i_global = li
                j_global = a_col_off + lj
                u_idx = i_global * m + j_global
                U[u_idx][col] += coeff
            
            # For V: B entries used
            # B global index: j * p + k, where j is row (0-3), k is col (0-5)
            # Local B indices (lj, lk) map to global (b_row_off + lj, b_col_off + lk)
            for (li, lk), coeff in zip(sf['b_indices'], sf['b_coeffs']):
                j_global = b_row_off + li
                k_global = b_col_off + lk
                v_idx = j_global * p + k_global
                V[v_idx][col] += coeff
            
            # For W: C entries
            # C is 2x6, C[i, k] = sum over j of A[i,j] * B[j,k]
            # The Strassen factor contributes to specific C positions.
            # We need to figure out which C positions this factor contributes to.
            
            # For a 2x2 * 2x2 multiplication with local indices (ci, ck) in {0,1}x{0,1}:
            # The global C position is (ci, b_col_off + ck)
            # C global index for W: k*n + i = (b_col_off + ck) * n + ci
            
            # Strassen contributions:
            # M1: C11 += 1, C22 += 1
            # M2: C12 += 1
            # M3: C21 += 1
            # M4: C11 -= 1, C12 += 1
            # M5: C21 += 1, C22 -= 1
            # M6: C11 += 1
            # M7: C11 += 1, C22 -= 1
            
            c_contribs = {
                'M1': [(0, 0, 1), (1, 1, 1)],
                'M2': [(0, 1, 1)],
                'M3': [(1, 0, 1)],
                'M4': [(0, 0, -1), (0, 1, 1)],
                'M5': [(1, 0, 1), (1, 1, -1)],
                'M6': [(0, 0, 1)],
                'M7': [(0, 0, 1), (1, 1, -1)],
            }
            
            for (ci, ck, coeff) in c_contribs[sf['c_position']]:
                i_global = ci
                k_global = b_col_off + ck
                w_idx = k_global * n + i_global
                W[w_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

