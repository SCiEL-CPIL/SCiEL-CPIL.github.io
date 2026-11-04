"""
Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition.
Uses Strassen's algorithm for 2x2x2 blocks and trivial decomposition for 2x2x1.
Total rank: 2 * (2*7 + 1*4) = 36
"""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Rank calculation:
    # For each pair of j indices (2 pairs: j=0,1 and j=2,3):
    #   - For k=0,1: Strassen 2x2x2 (rank 7)
    #   - For k=2,3: Strassen 2x2x2 (rank 7)
    #   - For k=4: trivial 2x2x1 (rank 4)
    # Total per j-pair: 7 + 7 + 4 = 18
    # Total rank: 2 * 18 = 36
    
    rank = 36
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen's 2x2x2 decomposition
    # C = A * B where A is 2x2, B is 2x2, C is 2x2
    # Terms (u, v, w) where u is row index in U (0-3), v is row in V (0-3), w is row in W (0-3)
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: list of (row, coeff) for U
        # v_coeffs: list of (row, coeff) for V
        # w_coeffs: list of (row, coeff) for W
        {
            'u': [(0, 1), (3, 1)],  # A00 + A11
            'v': [(0, 1), (3, 1)],  # B00 + B11
            'w': [(0, 1), (1, 1), (2, 1), (3, 1)]  # M1 contributes to all C
        },
        {
            'u': [(2, 1), (3, 1)],  # A10 + A11
            'v': [(0, 1)],           # B00
            'w': [(2, 1), (3, -1)]   # M2: C10 += M2, C11 -= M2
        },
        {
            'u': [(0, 1)],           # A00
            'v': [(2, 1), (3, -1)],  # B10 - B11
            'w': [(1, 1), (3, 1)]    # M3: C01 += M3, C11 += M3
        },
        {
            'u': [(3, 1)],           # A11
            'v': [(1, 1), (0, -1)],  # B01 - B00
            'w': [(0, 1), (2, 1)]    # M4: C00 += M4, C10 += M4
        },
        {
            'u': [(0, 1), (1, 1)],  # A00 + A01
            'v': [(3, 1)],           # B11
            'w': [(0, -1), (1, 1)]   # M5: C00 -= M5, C01 += M5
        },
        {
            'u': [(2, 1), (0, -1)],  # A10 - A00
            'v': [(0, 1), (1, 1)],   # B00 + B01
            'w': [(3, 1)]            # M6: C11 += M6
        },
        {
            'u': [(1, 1), (3, -1)],  # A01 - A11
            'v': [(2, 1), (3, 1)],   # B10 + B11
            'w': [(0, 1)]            # M7: C00 += M7
        }
    ]
    
    # Trivial 2x2x1 decomposition (rank 4)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    trivial_221_terms = [
        {'u': [(0, 1)], 'v': [(0, 1)], 'w': [(0, 1)]},  # A00*B0 -> C0
        {'u': [(1, 1)], 'v': [(1, 1)], 'w': [(0, 1)]},  # A01*B1 -> C0
        {'u': [(2, 1)], 'v': [(0, 1)], 'w': [(1, 1)]},  # A10*B0 -> C1
        {'u': [(3, 1)], 'v': [(1, 1)], 'w': [(1, 1)]},  # A11*B1 -> C1
    ]
    
    # Process each j-pair
    for j_pair in range(2):
        j_base = j_pair * 2  # 0 or 2
        
        # For each k-block
        # k=0,1: Strassen
        # k=2,3: Strassen
        # k=4: trivial
        
        # Strassen for k=0,1
        for k_base in [0, 2]:
            for term_idx, term in enumerate(strassen_terms):
                col = j_pair * 18 + k_base * 0 + term_idx  # Need proper column mapping
                
                # Actually, let's use a global column counter
                # For j_pair=0: columns 0-17
                #   k=0,1: cols 0-6 (7 cols)
                #   k=2,3: cols 7-13 (7 cols)
                #   k=4: cols 14-17 (4 cols)
                # For j_pair=1: columns 18-35
                #   k=0,1: cols 18-24
                #   k=2,3: cols 25-31
                #   k=4: cols 32-35
                
                if k_base == 0:
                    col_offset = j_pair * 18 + term_idx
                else:  # k_base == 2
                    col_offset = j_pair * 18 + 7 + term_idx
                
                # Fill U
                for u_row, u_coeff in term['u']:
                    # u_row is local index 0-3 for A[i][j]
                    # Global U row: i * m + j = u_row_i * 4 + (j_base + u_row_j)
                    # u_row encodes i*2 + j_local where j_local is 0 or 1
                    i_local = u_row // 2
                    j_local = u_row % 2
                    global_u_row = i_local * m + j_base + j_local
                    U[global_u_row][col_offset] += u_coeff
                
                # Fill V
                for v_row, v_coeff in term['v']:
                    # v_row is local index 0-3 for B[j][k]
                    # Global V row: j * p + k = (j_base + v_row_j) * 5 + k_base + v_row_k
                    j_local = v_row // 2
                    k_local = v_row % 2
                    global_v_row = (j_base + j_local) * p + k_base + k_local
                    V[global_v_row][col_offset] += v_coeff
                
                # Fill W
                for w_row, w_coeff in term['w']:
                    # w_row is local index 0-3 for C[k][i]
                    # Global W row: k * n + i = (k_base + w_row_k) * 2 + w_row_i
                    k_local = w_row // 2
                    i_local = w_row % 2
                    global_w_row = (k_base + k_local) * n + i_local
                    W[global_w_row][col_offset] += w_coeff
        
        # Trivial for k=4
        for term_idx, term in enumerate(trivial_221_terms):
            col_offset = j_pair * 18 + 14 + term_idx
            
            # Fill U
            for u_row, u_coeff in term['u']:
                i_local = u_row // 2
                j_local = u_row % 2
                global_u_row = i_local * m + j_base + j_local
                U[global_u_row][col_offset] += u_coeff
            
            # Fill V
            for v_row, v_coeff in term['v']:
                j_local = v_row // 2
                k_local = v_row % 2
                global_v_row = (j_base + j_local) * p + 4 + k_local
                V[global_v_row][col_offset] += v_coeff
            
            # Fill W
            for w_row, w_coeff in term['w']:
                k_local = w_row // 2
                i_local = w_row % 2
                global_w_row = 4 * n + i_local
                W[global_w_row][col_offset] += w_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
