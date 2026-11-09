# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    Uses 4+1 block partition with recursive Strassen for 4x4x4 block (rank 110).
    """
    n, m, p = DIMENSIONS
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Strassen 2x2x2 terms
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_2x2x2(I_map, J_map, K_map, u_scale=1.0, v_scale=1.0, w_scale=1.0):
        for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), val in u_coeffs.items():
                gi = I_map[li]
                gj = J_map[lj]
                u_entries.append((gi * m + gj, val * u_scale))
            for (lj, lk), val in v_coeffs.items():
                gj = J_map[lj]
                gk = K_map[lk]
                v_entries.append((gj * p + gk, val * v_scale))
            for (li, lk), val in w_coeffs.items():
                gi = I_map[li]
                gk = K_map[lk]
                w_entries.append((gk * n + gi, val * w_scale))
            add_column(u_entries, v_entries, w_entries)

    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )

    def add_4x4x4_strassen(i_list, j_list, k_list):
        """Add rank-49 Strassen decomposition for a 4x4x4 block."""
        # i_list, j_list, k_list are each lists of 4 indices
        # Partition into 2x2 sub-blocks
        # Sub-block indices: (0,0), (0,1), (1,0), (1,1)
        # For rows: i_list[0:2] is sub-block 0, i_list[2:4] is sub-block 1
        # For cols: j_list[0:2] is sub-block 0, j_list[2:4] is sub-block 1
        # For result: k_list[0:2] is sub-block 0, k_list[2:4] is sub-block 1
        
        # Top-level Strassen: 7 combinations
        # Each combination has coefficients for the 4 sub-blocks in each dimension
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Mi, we need to decompose the 2x2x2 product
        # The A part: combination of sub-blocks with coefficients
        # The B part: combination of sub-blocks with coefficients
        # The C part: combination of sub-blocks with coefficients
        
        # Define the 7 Strassen terms at the 4x4 level
        # Each term: (A_coeffs, B_coeffs, C_coeffs) where coeffs are dicts mapping (subrow, subcol) -> coeff
        STRASSEN_4x4_TOP = [
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            (
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1}
            ),
            # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
            (
                {(1,0): 1, (1,1): 1},
                {(0,0): 1},
                {(1,0): 1, (1,1): -1}
            ),
            # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
            (
                {(0,0): 1},
                {(0,1): 1, (1,1): -1},
                {(0,1): 1, (1,1): 1}
            ),
            # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
            (
                {(1,1): 1},
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (1,0): 1}
            ),
            # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
            (
                {(0,0): 1, (0,1): 1},
                {(1,1): 1},
                {(0,0): -1, (0,1): 1}
            ),
            # M6 = (A10-A00)(B00+B01) -> C11 += M6
            (
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (0,1): 1},
                {(1,1): 1}
            ),
            # M7 = (A01-A11)(B10+B11) -> C00 += M7
            (
                {(0,1): 1, (1,1): -1},
                {(1,0): 1, (1,1): 1},
                {(0,0): 1}
            ),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in STRASSEN_4x4_TOP:
            # For this top-level Strassen term, we need to decompose
            # the product of (sum of A sub-blocks) x (sum of B sub-blocks)
            # into the C sub-blocks
            
            # The A part is a linear combination of 2x2 sub-blocks
            # The B part is a linear combination of 2x2 sub-blocks
            # The C part is a linear combination of 2x2 sub-blocks
            
            # For each pair of A sub-block (ai, aj) and B sub-block (bi, bj),
            # we have a 2x2x2 product that needs Strassen decomposition
            # The coefficient is a_coeffs[(ai,aj)] * b_coeffs[(bi,bj)]
            # And it contributes to C sub-blocks with c_coeffs[(ci,cj)]
            
            # Actually, the structure is:
            # C = sum over (ci,cj) of c_coeffs[(ci,cj)] * (A_combined x B_combined)
            # where A_combined = sum over (ai,aj) of a_coeffs[(ai,aj)] * A[ai,aj]
            # and B_combined = sum over (bi,bj) of b_coeffs[(bi,bj)] * B[bi,bj]
            
            # But wait, that's not quite right. The Strassen formula is:
            # C = sum of c_coeffs * (A_combined * B_combined)
            # where A_combined and B_combined are specific linear combinations
            
            # For the tensor decomposition, each rank-1 term corresponds to:
            # u = specific A entry pattern, v = specific B entry pattern, w = specific C entry pattern
            
            # The 2x2x2 Strassen decomposition gives us 7 rank-1 terms for a 2x2x2 product.
            # Here, the "2x2" in each dimension refers to the sub-block indices.
            
            # So for each of the 7 Strassen 2x2x2 terms, we expand:
            # u: for each (li, lj) in the 2x2 pattern, with coefficient val_u
            #   this corresponds to global indices (i_list[2*li + si], j_list[2*lj + sj]) for si,sj in {0,1}
            #   Wait, no. The sub-block (li, lj) contains 4 elements: 
            #   (i_list[2*li + 0], j_list[2*lj + 0]), (i_list[2*li + 0], j_list[2*lj + 1]),
            #   (i_list[2*li + 1], j_list[2*lj + 0]), (i_list[2*li + 1], j_list[2*lj + 1])
            
            # Hmm, I need to think about this differently.
            
            # The 4x4x4 tensor is: T[i,j,k] = sum_l A[i,l] * B[l,k]
            # Partition i into {i0,i1,i2,i3}, j into {j0,j1,j2,j3}, k into {k0,k1,k2,k3}
            # Sub-block (0,0) for A: rows {i0,i1}, cols {j0,j1}
            # Sub-block (0,1) for A: rows {i0,i1}, cols {j2,j3}
            # Sub-block (1,0) for A: rows {i2,i3}, cols {j0,j1}
            # Sub-block (1,1) for A: rows {i2,i3}, cols {j2,j3}
            # Similarly for B and C.
            
            # The Strassen decomposition at the 4x4 level gives 7 terms.
            # Each term is a product of a linear combination of A sub-blocks and a linear combination of B sub-blocks,
            # contributing to a linear combination of C sub-blocks.
            
            # For each such term, the "2x2x2" product is between the sub-blocks.
            # We then apply Strassen to each 2x2x2 sub-product.
            
            # So for top-level Strassen term t with:
            # A_coeffs: dict from (subrow, subcol) to coeff
            # B_coeffs: dict from (subrow, subcol) to coeff  
            # C_coeffs: dict from (subrow, subcol) to coeff
            
            # The contribution is: sum_{(ci,cj)} C_coeffs[(ci,cj)] * (A_comb @ B_comb)
            # where A_comb = sum_{(ai,aj)} A_coeffs[(ai,aj)] * A_subblock[ai,aj]
            # and B_comb = sum_{(bi,bj)} B_coeffs[(bi,bj)] * B_subblock[bi,bj]
            
            # But for the tensor decomposition, we need to express this as sum of rank-1 terms.
            # The 2x2x2 Strassen decomposition tells us how to decompose a 2x2x2 matrix product.
            # Here, the "entries" of the 2x2 matrices are themselves 2x2 blocks.
            
            # So for each of the 7 Strassen 2x2x2 terms (with patterns u_pat, v_pat, w_pat):
            # u_pat: dict from (li, lj) to val  [li, lj are sub-block indices]
            # v_pat: dict from (lj, lk) to val  [lj, lk are sub-block indices]
            # w_pat: dict from (li, lk) to val  [li, lk are sub-block indices]
            
            # The contribution to the tensor is:
            # For each (li, lj) in u_pat with value vu:
            #   For each (lj2, lk) in v_pat with value vv:
            #     (li, lj) and (lj2, lk) must share the same "middle" index
            #     Actually, in the 2x2x2 Strassen, the indices are:
            #     u: (row, col) in {0,1}x{0,1}
            #     v: (col, row) in {0,1}x{0,1}  [note: col of A = row of B]
            #     w: (row, col) in {0,1}x{0,1}
            #     And the constraint is that u's col = v's row for the product to make sense.
            
            # Wait, I'm overcomplicating this. Let me re-examine the Strassen 2x2x2 decomposition.
            
            # In the standard Strassen 2x2x2:
            # A = [[A00, A01], [A10, A11]]
            # B = [[B00, B01], [B10, B11]]
            # C = A @ B = [[C00, C01], [C10, C11]]
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # For the tensor decomposition, each Mi is a 2x2x2 tensor (since Aij and Bjk are scalars in the 2x2 case, but here they are 2x2 blocks).
            
            # Actually, in our case for the 4x4x4 decomposition:
            # The "entries" of the 2x2 matrices (at the top level) are 2x2 blocks.
            # So M1 = (A00+A11)(B00+B11) is a 2x2x2 tensor where:
            # - The A part has entries at sub-blocks (0,0) and (1,1) with coefficient 1
            # - The B part has entries at sub-blocks (0,0) and (1,1) with coefficient 1
            # - The C part has entries at sub-blocks (0,0) and (1,1) with coefficient 1
            
            # And this 2x2x2 tensor (at the sub-block level) needs to be decomposed into rank-1 terms.
            # Each rank-1 term at the sub-block level corresponds to a 2x2x2 Strassen term at the element level.
            
            # So the total rank is 7 (top-level) * 7 (bottom-level) = 49.
            
            # For each top-level Strassen term (a_coeffs, b_coeffs, c_coeffs):
            # For each bottom-level Strassen term (u_pat, v_pat, w_pat):
            #   The rank-1 term has:
            #   u: for each (li, lj) in u_pat with value vu:
            #        global A index: (i_list[2*li + si], j_list[2*lj + sj]) for si,sj in {0,1}
            #        coefficient: vu * a_coeffs[(li, lj)]
            #   v: for each (lj, lk) in v_pat with value vv:
            #        global B index: (j_list[2*lj + si], k_list[2*lk + sj]) for si,sj in {0,1}
            #        coefficient: vv * b_coeffs[(lj, lk)]
            #   w: for each (li, lk) in w_pat with value vw:
            #        global C index: (k_list[2*lk + sj], i_list[2*li + si]) for si,sj in {0,1}
            #        coefficient: vw * c_coeffs[(li, lk)]
            
            # Wait, I need to be more careful with the indices.
            
            # Let me define:
            # i_list = [i0, i1, i2, i3] where i0,i1 are sub-block 0 and i2,i3 are sub-block 1
            # j_list = [j0, j1, j2, j3] where j0,j1 are sub-block 0 and j2,j3 are sub-block 1
            # k_list = [k0, k1, k2, k3] where k0,k1 are sub-block 0 and k2,k3 are sub-block 1
            
            # Sub-block (li, lj) of A contains elements:
            # (i_list[2*li + si], j_list[2*lj + sj]) for si in {0,1}, sj in {0,1}
            
            # Sub-block (lj, lk) of B contains elements:
            # (j_list[2*lj + si], k_list[2*lk + sj]) for si in {0,1}, sj in {0,1}
            
            # Sub-block (li, lk) of C contains elements:
            # (k_list[2*lk + sj], i_list[2*li + si]) for si in {0,1}, sj in {0,1}
            
            # For a given top-level term (a_coeffs, b_coeffs, c_coeffs) and bottom-level term (u_pat, v_pat, w_pat):
            # u_entries: for each (li, lj) in u_pat with value vu:
            #   for si in {0,1}, sj in {0,1}:
            #     if (li, lj) in a_coeffs:
            #       global_idx = i_list[2*li + si] * m + j_list[2*lj + sj]
            #       coeff = vu * a_coeffs[(li, lj)]
            #       add (global_idx, coeff)
            # v_entries: for each (lj, lk) in v_pat with value vv:
            #   for si in {0,1}, sj in {0,1}:
            #     if (lj, lk) in b_coeffs:
            #       global_idx = j_list[2*lj + si] * p + k_list[2*lk + sj]
            #       coeff = vv * b_coeffs[(lj, lk)]
            #       add (global_idx, coeff)
            # w_entries: for each (li, lk) in w_pat with value vw:
            #   for si in {0,1}, sj in {0,1}:
            #     if (li, lk) in c_coeffs:
            #       global_idx = k_list[2*lk + sj] * n + i_list[2*li + si]
            #       coeff = vw * c_coeffs[(li, lk)]
            #       add (global_idx, coeff)
            
            for u_pat, v_pat, w_pat in STRASSEN_2x2_TERMS:
                u_entries = []
                v_entries = []
                w_entries = []
                for (li, lj), vu in u_pat.items():
                    if (li, lj) in a_coeffs:
                        for si in range(2):
                            for sj in range(2):
                                gi = i_list[2*li + si]
                                gj = j_list[2*lj + sj]
                                u_entries.append((gi * m + gj, vu * a_coeffs[(li, lj)]))
                for (lj, lk), vv in v_pat.items():
                    if (lj, lk) in b_coeffs:
                        for si in range(2):
                            for sj in range(2):
                                gj = j_list[2*lj + si]
                                gk = k_list[2*lk + sj]
                                v_entries.append((gj * p + gk, vv * b_coeffs[(lj, lk)]))
                for (li, lk), vw in w_pat.items():
                    if (li, lk) in c_coeffs:
                        for si in range(2):
                            for sj in range(2):
                                gi = i_list[2*li + si]
                                gk = k_list[2*lk + sj]
                                w_entries.append((gk * n + gi, vw * c_coeffs[(li, lk)]))
                add_column(u_entries, v_entries, w_entries)

    def add_block(i_list, j_list, k_list):
        ni, nj, nk = len(i_list), len(j_list), len(k_list)
        
        if ni == 4 and nj == 4 and nk == 4:
            add_4x4x4_strassen(i_list, j_list, k_list)
        elif ni == 2 and nj == 2 and nk == 2:
            add_strassen_2x2x2(
                {0: i_list[0], 1: i_list[1]},
                {0: j_list[0], 1: j_list[1]},
                {0: k_list[0], 1: k_list[1]}
            )
        elif ni == 1 and nj == 1 and nk == 1:
            schoolbook_block(i_list, j_list, k_list)
        elif ni == 4 and nj == 4 and nk == 1:
            # 4x4x1: rank 16
            schoolbook_block(i_list, j_list, k_list)
        elif ni == 4 and nj == 1 and nk == 4:
            # 4x1x4: rank 16
            schoolbook_block(i_list, j_list, k_list)
        elif ni == 1 and nj == 4 and nk == 4:
            # 1x4x4: rank 16
            schoolbook_block(i_list, j_list, k_list)
        elif ni == 4 and nj == 1 and nk == 1:
            # 4x1x1: rank 4
            schoolbook_block(i_list, j_list, k_list)
        elif ni == 1 and nj == 4 and nk == 1:
            # 1x4x1: rank 4
            schoolbook_block(i_list, j_list, k_list)
        elif ni == 1 and nj == 1 and nk == 4:
            # 1x1x4: rank 4
            schoolbook_block(i_list, j_list, k_list)
        else:
            schoolbook_block(i_list, j_list, k_list)

    # Partition 4+1
    parts = [[0, 1, 2, 3], [4]]
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                i_list = parts[bi]
                j_list = parts[bj]
                k_list = parts[bk]
                add_block(i_list, j_list, k_list)

    rank = len(columns)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
