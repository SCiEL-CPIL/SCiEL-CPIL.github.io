"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Uses Strassen's algorithm recursively to reduce rank.
    Rank = 49 (for 4x4x4 part) + 49 (for 4x4x3 part, padded to 4x4x4) = 98.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # We split B into B1 (4x4) and B2 (4x3).
    # We compute A*B1 and A*B2 separately.
    # A*B1 is 4x4x4 multiplication. Rank 49.
    # A*B2 is 4x4x3 multiplication. We pad B2 to 4x4 with zeros. Rank 49.
    # Total rank 98.
    
    rank_total = 98
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank_total for _ in range(m * p)]  # 28 x 98
    W = [[0] * rank_total for _ in range(p * n)]  # 28 x 98
    
    # Helper to get Strassen factors for 2x2x2
    def get_strassen_2x2x2():
        # Returns U, V, W of shape (4, 7)
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = 1
        
        # Term 2: M2 = (A10+A11)B00
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[1][1] = 1; w[3][1] = -1
        
        # Term 3: M3 = A00(B01-B11)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[2][2] = 1; w[3][2] = 1
        
        # Term 4: M4 = A11(B10-B00)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1; w[1][3] = 1
        
        # Term 5: M5 = (A00+A01)B11
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1; w[2][4] = 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[0][6] = 1; w[3][6] = 1
        
        return u, v, w

    u_base, v_base, w_base = get_strassen_2x2x2()
    
    # Define the 7 terms for 4x4x4 Strassen
    # Each term is a linear combination of A blocks and B blocks.
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    # Global A index: i*m + j = i*4 + j
    # Global B index: j*p + k = j*4 + k (for the 4x4 part)
    # Global C index: k*n + i = k*4 + i
    
    # We will iterate through the 7 Strassen terms for the 4x4 level.
    # For each term, we identify which A blocks and B blocks are involved.
    # Then we use the base 2x2x2 factors to populate the global U, V, W.
    
    col_offset = 0
    
    # Term 1: (A00+A11)(B00+B11)
    # A blocks: 00, 11. B blocks: 00, 11.
    # We need to sum the contributions.
    # Since Strassen is linear, we can just add the factors.
    # However, the factors for (A00+A11) are not just the sum of factors for A00 and A11?
    # Yes, U is linear in A. So U for (A00+A11) is U(A00) + U(A11).
    # But we are constructing a rank decomposition. We can't just add ranks.
    # We must use the specific linear combinations defined in the 2x2x2 Strassen.
    # Wait, the 2x2x2 Strassen already handles the linear combinations of the 2x2 blocks.
    # So for the 4x4 level, we treat each 2x2 block as a scalar in the 2x2x2 Strassen?
    # No, that's not right.
    # The 4x4 Strassen is:
    # M1 = (A00+A11)(B00+B11)
    # This is a product of two 2x2 matrices.
    # We can decompose this product using the 2x2x2 Strassen factors.
    # So we take the 2x2x2 factors for A=(A00+A11) and B=(B00+B11).
    # The U part for A=(A00+A11) will have entries in the rows corresponding to A00 and A11.
    # The V part for B=(B00+B11) will have entries in the rows corresponding to B00 and B11.
    # The W part will be in the rows corresponding to C00 and C11.
    
    # Let's define a helper to add a 2x2x2 Strassen decomposition to the global matrices.
    # It takes:
    # - a_list: list of (block_row, block_col, coeff) for A
    # - b_list: list of (block_row, block_col, coeff) for B
    # - c_list: list of (block_row, block_col, coeff) for C
    # - col_start: starting column in U, V, W
    
    def add_2x2x2_strassen(a_list, b_list, c_list, col_start):
        nonlocal U, V, W
        for t in range(7):
            # For each of the 7 terms in the base Strassen
            for i_loc in range(4):
                if u_base[i_loc][t] != 0:
                    # i_loc maps to local A index
                    # We need to map this to the global A index based on a_list
                    # This is tricky because u_base[i_loc][t] is a coefficient for a specific local A element.
                    # We need to distribute this coefficient to the global A elements specified in a_list.
                    # But a_list describes the linear combination of blocks.
                    # Actually, the input to the base Strassen is a 2x2 matrix A_local.
                    # A_local is formed by the linear combination of the global blocks.
                    # So if A_local[i_loc//2, i_loc%2] has coefficient c, and A_local is sum of blocks...
                    # This is getting complicated.
                    
                    # Simpler: Just hardcode the 7 terms for 4x4x4 and use the 2x2x2 factors directly.
                    pass
        # This approach is too complex to generalize dynamically.
        # Let's just generate the 49 terms explicitly.
        
    # Alternative: Just use the schoolbook for the 2x2 blocks inside the Strassen terms?
    # No, that would be 8 * 7 = 56 for the 4x4 part.
    # We want 49.
    
    # Let's manually construct the 49 terms.
    # We know the 7 Strassen terms for 4x4.
    # For each term, we use the 7 Strassen terms for 2x2.
    # Total 49 terms.
    
    # We will create a list of "sub-decompositions".
    # Each sub-decomposition is a 2x2x2 Strassen instance.
    # We need to map the local indices of the 2x2x2 instance to the global 4x4x4 indices.
    
    # Let's define the 7 high-level terms again.
    # H1: (A00+A11)(B00+B11) -> C00, C11
    # H2: (A10+A11)B00 -> C10, C11
    # H3: A00(B01-B11) -> C01, C11
    # H4: A11(B10-B00) -> C00, C10
    # H5: (A00+A01)B11 -> C00, C01
    # H6: (A10-A00)(B00+B01) -> C11
    # H7: (A01-A11)(B10+B11) -> C00, C11
    
    # For each high-level term, we instantiate a 2x2x2 Strassen.
    # The "A" of the 2x2x2 is the linear combination of 2x2 blocks.
    # The "B" of the 2x2x2 is the linear combination of 2x2 blocks.
    # The "C" of the 2x2x2 is the linear combination of 2x2 blocks.
    
    # We need a function that, given a linear combination of A blocks, B blocks, and C blocks,
    # and the base 2x2x2 factors, adds to the global U, V, W.
    
    def add_instance(a_comb, b_comb, c_comb, col_start):
        # a_comb: list of (br, bc, coeff)
        # b_comb: list of (br, bc, coeff)
        # c_comb: list of (br, bc, coeff)
        for t in range(7):
            for i_loc in range(4):
                coeff_u = u_base[i_loc][t]
                if coeff_u != 0:
                    # i_loc is index in local A (2x2).
                    # local row = i_loc // 2, local col = i_loc % 2.
                    l_row = i_loc // 2
                    l_col = i_loc % 2
                    # We need to find which global blocks contribute to this local element.
                    # Actually, the local A is formed by a_comb.
                    # The local A element (l_row, l_col) is the sum of coeff * A_block[br, bc] where the block's (l_row, l_col) element is taken.
                    # Wait, no. The blocks are 2x2.
                    # A_local is a 2x2 matrix.
                    # A_local[i, j] = sum_{(br, bc, c) in a_comb} c * A_global[2*br+i, 2*bc+j]
                    # So for a fixed local (i, j), we add coeff_u * c to U[global_i, global_j, col_start+t]
                    # where global_i = 2*br+i, global_j = 2*bc+j.
                    
                    for br, bc, c in a_comb:
                        g_i = 2 * br + l_row
                        g_j = 2 * bc + l_col
                        U[g_i * 4 + g_j][col_start + t] += coeff_u * c
                        
            for j_loc in range(4):
                coeff_v = v_base[j_loc][t]
                if coeff_v != 0:
                    l_row = j_loc // 2
                    l_col = j_loc % 2
                    for br, bc, c in b_comb:
                        g_i = 2 * br + l_row
                        g_j = 2 * bc + l_col
                        V[g_i * 4 + g_j][col_start + t] += coeff_v * c
                        
            for k_loc in range(4):
                coeff_w = w_base[k_loc][t]
                if coeff_w != 0:
                    l_row = k_loc // 2
                    l_col = k_loc % 2
                    for br, bc, c in c_comb:
                        g_i = 2 * br + l_row
                        g_j = 2 * bc + l_col
                        # C index is k*n + i. Here n=4.
                        # g_i is row of C, g_j is col of C.
                        # So index is g_j * 4 + g_i.
                        W[g_j * 4 + g_i][col_start + t] += coeff_w * c

    # High Level Term 1: (A00+A11)(B00+B11)
    # A_comb: [(0,0,1), (1,1,1)]
    # B_comb: [(0,0,1), (1,1,1)]
    # C_comb: [(0,0,1), (1,1,1)]
    add_instance([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 0)
    
    # High Level Term 2: (A10+A11)B00
    # A_comb: [(1,0,1), (1,1,1)]
    # B_comb: [(0,0,1)]
    # C_comb: [(1,0,1), (1,1,-1)]
    add_instance([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)], 7)
    
    # High Level Term 3: A00(B01-B11)
    # A_comb: [(0,0,1)]
    # B_comb: [(0,1,1), (1,1,-1)]
    # C_comb: [(0,1,1), (1,1,1)]
    add_instance([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], 14)
    
    # High Level Term 4: A11(B10-B00)
    # A_comb: [(1,1,1)]
    # B_comb: [(1,0,1), (0,0,-1)]
    # C_comb: [(0,0,1), (1,0,1)]
    add_instance([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], 21)
    
    # High Level Term 5: (A00+A01)B11
    # A_comb: [(0,0,1), (0,1,1)]
    # B_comb: [(1,1,1)]
    # C_comb: [(0,0,-1), (0,1,1)]
    add_instance([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)], 28)
    
    # High Level Term 6: (A10-A00)(B00+B01)
    # A_comb: [(1,0,1), (0,0,-1)]
    # B_comb: [(0,0,1), (0,1,1)]
    # C_comb: [(1,1,1)]
    add_instance([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], 35)
    
    # High Level Term 7: (A01-A11)(B10+B11)
    # A_comb: [(0,1,1), (1,1,-1)]
    # B_comb: [(1,0,1), (1,1,1)]
    # C_comb: [(0,0,1), (1,1,1)]
    add_instance([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 42)
    
    # Now for the second part: A * B2 (4x4 * 4x3)
    # We pad B2 to 4x4 with zeros in the last column.
    # B2_pad = [B2 | 0]
    # This is a 4x4 matrix.
    # We use the same Strassen decomposition.
    # The B blocks for this part are:
    # B_block_00: rows 0-1, cols 0-1 of B2_pad (which are cols 4-5 of B)
    # B_block_01: rows 0-1, cols 2-3 of B2_pad (which are cols 6-7 of B, but col 7 is 0)
    # B_block_10: rows 2-3, cols 0-1 of B2_pad (which are cols 4-5 of B)
    # B_block_11: rows 2-3, cols 2-3 of B2_pad (which are cols 6-7 of B, but col 7 is 0)
    
    # We need to map the local B indices of the 4x4 Strassen to the global B indices (4x7).
    # Global B index: j*7 + k.
    # In the previous part, we used j*4 + k for the 4x4 B.
    # Here, we use j*7 + k.
    
    # We will modify the add_instance function to accept a mapping for B indices.
    # Or just write a separate function.
    
    def add_instance_padded(a_comb, b_comb, c_comb, col_start):
        for t in range(7):
            for i_loc in range(4):
                coeff_u = u_base[i_loc][t]
                if coeff_u != 0:
                    l_row = i_loc // 2
                    l_col = i_loc % 2
                    for br, bc, c in a_comb:
                        g_i = 2 * br + l_row
                        g_j = 2 * bc + l_col
                        U[g_i * 4 + g_j][col_start + t] += coeff_u * c
                        
            for j_loc in range(4):
                coeff_v = v_base[j_loc][t]
                if coeff_v != 0:
                    l_row = j_loc // 2
                    l_col = j_loc % 2
                    for br, bc, c in b_comb:
                        g_i = 2 * br + l_row
                        g_j = 2 * bc + l_col
                        # Map local B (4x4) to global B (4x7)
                        # Local B col l_col corresponds to global B col:
                        # If bc=0, global cols are 4, 5.
                        # If bc=1, global cols are 6, 7 (but 7 is 0).
                        # So global_k = 4 + (2*bc + l_col)
                        g_k = 4 + (2 * bc + l_col)
                        # Global B index: g_i * 7 + g_k
                        V[g_i * 7 + g_k][col_start + t] += coeff_v * c
                        
            for k_loc in range(4):
                coeff_w = w_base[k_loc][t]
                if coeff_w != 0:
                    l_row = k_loc // 2
                    l_col = k_loc % 2
                    for br, bc, c in c_comb:
                        g_i = 2 * br + l_row
                        g_j = 2 * bc + l_col
                        # C index is g_j * 4 + g_i
                        # But C is 4x7.
                        # The C blocks are 2x2.
                        # C_block (br, bc) corresponds to rows 2*br..2*br+1, cols 4+2*bc..4+2*bc+1.
                        # So global_k = 4 + (2*bc + l_col)
                        # global_i = 2*br + l_row
                        g_k = 4 + (2 * bc + l_col)
                        g_i_global = 2 * br + l_row
                        W[g_k * 4 + g_i_global][col_start + t] += coeff_w * c

    # The A_comb and C_comb are the same as before.
    # The B_comb is the same structure, but the mapping is different.
    
    add_instance_padded([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 49)
    add_instance_padded([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)], 56)
    add_instance_padded([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], 63)
    add_instance_padded([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], 70)
    add_instance_padded([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)], 77)
    add_instance_padded([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], 84)
    add_instance_padded([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 91)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
