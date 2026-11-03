"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # u, v, w are length 4 vectors (flattened 2x2)
    # Flattening order: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
    ]
    
    # Recursive Strassen for 4x4x4
    # We define 7 intermediate products M1..M7
    # Each Mk is a product of two 2x2 matrices (linear combinations of blocks)
    # We express the 4x4x4 tensor as sum of 7 * 7 = 49 rank-1 terms
    
    # Helper to get global indices for a 2x2 block in a 4x4 matrix
    def get_block_indices_4x4(block_i, block_j):
        start_r = block_i * 2
        start_c = block_j * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = start_r + dr
                c = start_c + dc
                indices.append(r * 4 + c)
        return indices

    # Precompute indices for 4x4 blocks
    block_idx_4x4 = {}
    for bi in range(2):
        for bj in range(2):
            block_idx_4x4[(bi, bj)] = get_block_indices_4x4(bi, bj)

    # Strassen combinations for 4x4x4
    # Each entry: (u_blocks, v_blocks, w_blocks)
    # u_blocks: list of (block_idx, coeff) for A
    # v_blocks: list of (block_idx, coeff) for B
    # w_blocks: list of (block_idx, coeff) for C
    
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C01 += -M1, C10 += -M1, C11 += M1
    # Wait, standard Strassen reconstruction:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # Let's define the 7 M's in terms of A and B blocks
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    strassen_4x4_combos = [
        # (A_factors, B_factors)
        # A_factors: dict {block: coeff}
        # B_factors: dict {block: coeff}
        (
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}
        ), # M1
        (
            {(1,0): 1, (1,1): 1}, 
            {(0,0): 1}
        ), # M2
        (
            {(0,0): 1}, 
            {(0,1): 1, (1,1): -1}
        ), # M3
        (
            {(1,1): 1}, 
            {(1,0): 1, (0,0): -1}
        ), # M4
        (
            {(0,0): 1, (0,1): 1}, 
            {(1,1): 1}
        ), # M5
        (
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (0,1): 1}
        ), # M6
        (
            {(0,1): 1, (1,1): -1}, 
            {(1,0): 1, (1,1): 1}
        ), # M7
    ]
    
    # Reconstruction coefficients for C blocks
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    c_recon = {
        (0,0): [(0, 1), (3, -1), (4, 1), (6, -1)], # M1, -M4, M5, -M7
        (0,1): [(2, 1), (6, 1)],                   # M3, M7
        (1,0): [(1, 1), (3, 1)],                   # M2, M4
        (1,1): [(0, 1), (2, -1), (5, 1), (6, 1)]   # M1, -M3, M6, M7
    }
    
    # Total rank for one 4x4x4 is 49.
    # We have two 4x4x4 multiplications (Left and Right halves of B).
    total_rank = 98
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 16 x 98
    V = [[0.0] * total_rank for _ in range(m * p)]  # 32 x 98
    W = [[0.0] * total_rank for _ in range(p * n)]  # 32 x 98
    
    # We will fill columns 0-48 for Left half (B_L)
    # and columns 49-97 for Right half (B_R)
    
    def fill_half(base_col, b_col_offset):
        # b_col_offset is 0 for Left, 4 for Right
        for m_idx, (a_factors, b_factors) in enumerate(strassen_4x4_combos):
            # For this M_m, we decompose the 2x2x2 product using strassen_2x2
            # The 2x2x2 product is between X and Y
            # X is a 2x2 matrix formed by linear combo of A blocks
            # Y is a 2x2 matrix formed by linear combo of B blocks
            
            # We need to construct the U, V, W vectors for this specific 2x2x2 multiplication
            # But the "inputs" to this 2x2x2 multiplication are not simple blocks, they are sums.
            # However, the linearity allows us to just add the contributions.
            
            # Let's iterate over the 7 Strassen terms for the inner 2x2x2
            for s_idx, (u_2, v_2, w_2) in enumerate(strassen_2x2):
                col = base_col + m_idx * 7 + s_idx
                
                # u_2, v_2, w_2 are coefficients for the elements of the 2x2 matrices X, Y, Z
                # X = sum_{(bi,bj) in a_factors} a_factors[(bi,bj)] * A_{bi,bj}
                # Y = sum_{(bi,bj) in b_factors} b_factors[(bi,bj)] * B_{bi,bj}
                
                # The u_2 vector corresponds to the coefficients of X's elements in the rank-1 term.
                # Wait, the standard Strassen 2x2x2 decomposition gives:
                # sum_k u_k v_k w_k = tensor
                # Here, the "tensor" is the multiplication of two 2x2 matrices.
                # But our "matrices" X and Y are themselves linear combinations of A and B blocks.
                
                # Let's expand:
                # Term_k = (sum_{i in A_blocks} c_ik X_i) * (sum_{j in B_blocks} d_jk Y_j) * w_k
                # This is not quite right.
                
                # Let's look at the structure.
                # We are computing C = A B.
                # We decomposed A B into sum_{m=1}^7 M_m.
                # Each M_m = X_m Y_m.
                # We decompose X_m Y_m using Strassen 2x2x2.
                # X_m Y_m = sum_{s=1}^7 u_{ms} v_{ms} w_{ms}.
                # Here u_{ms}, v_{ms}, w_{ms} are 2x2 matrices (flattened).
                
                # So the total contribution to C is:
                # sum_m sum_s (sum_{a_blk} c_{m,a} A_a) * (sum_{b_blk} d_{m,b} B_b) * w_{ms} ... no.
                
                # The u, v, w in strassen_2x2 are scalars multiplying the elements of the 2x2 matrices.
                # Let X be the 2x2 matrix. Its elements are linear combos of A blocks.
                # X_{ij} = sum_{(bi,bj) in a_factors} a_factors[(bi,bj)] * (A_{bi,bj})_{ij}
                
                # The rank-1 term for the inner product is:
                # sum_{r,c} u_{rc} X_{rc} * sum_{r',c'} v_{r'c'} Y_{r'c'} * w_{r''c''} ... no.
                
                # The tensor for 2x2x2 multiplication is:
                # T_{i,j,k} = sum_l A_{il} B_{lj}
                # Strassen: T = sum_s u_s v_s w_s
                # where u_s, v_s, w_s are vectors of length 4.
                # u_s is coefficients for A elements.
                # v_s is coefficients for B elements.
                # w_s is coefficients for C elements.
                
                # So, for a fixed m and s:
                # The contribution to C is:
                # sum_{element a in 2x2} u_{s,a} * X_a * (sum_{element b in 2x2} v_{s,b} * Y_b) * w_{s,c} C_c ?
                
                # No, the decomposition is:
                # C_{rc} = sum_l A_{rl} B_{lc}
                # Using Strassen:
                # C = sum_s (sum_a u_{s,a} A_a) (sum_b v_{s,b} B_b) w_s ? No.
                
                # The identity is:
                # sum_{a,b,c} A_a B_b C_c = sum_s (sum_a u_{s,a} A_a) (sum_b v_{s,b} B_b) (sum_c w_{s,c} C_c) ? No.
                
                # It is:
                # sum_{i,j,k} A_{ij} B_{jk} C_{ki} = sum_s (sum_{ij} u_{s,ij} A_{ij}) (sum_{jk} v_{s,jk} B_{jk}) (sum_{ki} w_{s,ki} C_{ki})
                
                # So, for the inner 2x2x2:
                # The "A" matrix is X_m.
                # X_m elements are linear combos of A blocks.
                # X_m_{rc} = sum_{(bi,bj) in a_factors} a_factors[(bi,bj)] * (A_{bi,bj})_{rc}
                
                # So the term (sum_{rc} u_{s,rc} X_m_{rc}) becomes:
                # sum_{rc} u_{s,rc} (sum_{(bi,bj)} a_factors[(bi,bj)] A_{bi,bj,rc})
                # = sum_{(bi,bj)} a_factors[(bi,bj)] * (sum_{rc} u_{s,rc} A_{bi,bj,rc})
                
                # This means we can distribute the coefficients u_{s,rc} to the global U vector.
                
                # So for each (bi,bj) in a_factors with coeff alpha:
                # We add alpha * u_{s,rc} to U[global_idx(bi,bj,rc)]
                
                # Similarly for V and B.
                
                # And for W, the w_{s,rc} applies to the C block resulting from this M_m.
                # But wait, M_m contributes to multiple C blocks with different signs.
                # We handled that in c_recon.
                # So for a given M_m, we need to add the contribution to each C block (ci,cj)
                # with weight c_recon[(ci,cj)][m_idx].
                
                # So, for fixed m, s:
                # 1. Update U:
                #    For each (bi,bj) in a_factors with coeff alpha:
                #       For each rc in 0..3:
                #          U[global_A(bi,bj,rc)][col] += alpha * u_2[rc]
                
                # 2. Update V:
                #    For each (bi,bj) in b_factors with coeff beta:
                #       For each rc in 0..3:
                #          V[global_B(bi,bj,rc, b_col_offset)][col] += beta * v_2[rc]
                
                # 3. Update W:
                #    For each (ci,cj) in c_recon keys:
                #       weight = c_recon[(ci,cj)][m_idx]
                #       For each rc in 0..3:
                #          W[global_C(ci,cj,rc, b_col_offset)][col] += weight * w_2[rc]
                
                # Let's implement this.
                
                # 1. U
                for (bi, bj), alpha in a_factors.items():
                    indices = block_idx_4x4[(bi, bj)]
                    for k, idx in enumerate(indices):
                        U[idx][col] += alpha * u_2[k]
                        
                # 2. V
                for (bi, bj), beta in b_factors.items():
                    indices = block_idx_4x4[(bi, bj)]
                    for k, idx in enumerate(indices):
                        # idx is local 4x4 index (r*4+c)
                        b_row = idx // 4
                        b_col_local = idx % 4
                        b_col_global = b_col_offset + b_col_local
                        v_flat_idx = b_row * p + b_col_global
                        V[v_flat_idx][col] += beta * v_2[k]
                        
                # 3. W
                for (ci, cj), recon_list in c_recon.items():
                    # Find weight for m_idx
                    weight = 0
                    for (m_i, w) in recon_list:
                        if m_i == m_idx:
                            weight = w
                            break
                    if weight != 0:
                        indices = block_idx_4x4[(ci, cj)]
                        for k, idx in enumerate(indices):
                            # idx is local 4x4 index (r*4+c)
                            c_row = idx // 4
                            c_col_local = idx % 4
                            c_col_global = b_col_offset + c_col_local
                            # W index: k * n + i
                            # k is col, i is row
                            w_flat_idx = c_col_global * n + c_row
                            W[w_flat_idx][col] += weight * w_2[k]
                        
    # Fill Left half (cols 0-3 of B, cols 0-3 of C)
    fill_half(0, 0)
    
    # Fill Right half (cols 4-7 of B, cols 4-7 of C)
    fill_half(49, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
