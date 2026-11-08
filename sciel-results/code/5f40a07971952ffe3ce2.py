"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-117 exact tensor decomposition for 5x5x5 matrix multiplication
    using block decomposition with Strassen for 2x2x2 sub-blocks.
    
    Split 5 into 2+2+1. Use Strassen rank-7 for 2x2x2 blocks,
    natural rank for smaller blocks. Total rank: 117.
    """
    n, m, p = DIMENSIONS  # all 5
    
    # Block group boundaries: group 0 = indices [0,1], group 1 = indices [2,3], group 2 = index [4]
    group_bounds = [(0, 2), (2, 4), (4, 5)]
    group_sizes = [2, 2, 1]
    
    def local_to_global(idx, group):
        """Convert local index (0-based within group) to global index."""
        start = group_bounds[group][0]
        return start + idx
    
    def build_U_block(local_entries, row_group, col_group):
        """Build a length-25 vector for U from local entries in a block."""
        vec = [0] * 25
        for (li, lj, val) in local_entries:
            gi = local_to_global(li, row_group)
            gj = local_to_global(lj, col_group)
            vec[gi * 5 + gj] = val
        return vec
    
    def build_V_block(local_entries, row_group, col_group):
        """Build a length-25 vector for V from local entries in a block."""
        vec = [0] * 25
        for (li, lj, val) in local_entries:
            gi = local_to_global(li, row_group)
            gj = local_to_global(lj, col_group)
            vec[gi * 5 + gj] = val
        return vec
    
    def build_W_block(local_entries, row_group, col_group):
        """Build a length-25 vector for W from local entries in a block."""
        vec = [0] * 25
        for (li, lj, val) in local_entries:
            gi = local_to_global(li, row_group)
            gj = local_to_global(lj, col_group)
            vec[gi * 5 + gj] = val
        return vec
    
    # Strassen's rank-7 decomposition for 2x2x2
    # Each term: (A_entries, B_entries, C_entries)
    # A_entries: list of (local_i, local_j, value) for the 2x2 A sub-matrix
    # B_entries: list of (local_j, local_k, value) for the 2x2 B sub-matrix  
    # C_entries: list of (local_k, local_i, value) for the 2x2 C sub-matrix (note: W is indexed by (k,i))
    
    strassen_2x2 = [
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        ([[(0,0,1),(1,1,1)]], [[(0,0,1),(1,1,1)]], [[(0,0,1),(1,1,1)]]),
        # M2: (A10+A11)B00 -> C10
        ([[(1,0,1),(1,1,1)]], [[(0,0,1)]], [[(0,1,1)]]),
        # M3: A00(B10-B11) -> C01
        ([[(0,0,1)]], [[(1,0,1),(1,1,-1)]], [[(1,0,1)]]),
        # M4: A11(B01-B00) -> C10
        ([[(1,1,1)]], [[(0,1,1),(0,0,-1)]], [[(0,1,1)]]),
        # M5: (A00+A01)B11 -> C00 - C11
        ([[(0,0,1),(0,1,1)]], [[(1,1,1)]], [[(0,0,1),(1,1,-1)]]),
        # M6: (A10-A00)(B00+B10) -> C11
        ([[(1,0,1),(0,0,-1)]], [[(0,0,1),(1,0,1)]], [[(1,1,1)]]),
        # M7: (A01-A11)(B01+B11) -> C00 + C11
        ([[(0,1,1),(1,1,-1)]], [[(0,1,1),(1,1,1)]], [[(0,0,1),(1,1,1)]]),
    ]
    
    terms = []
    
    for ga in range(3):
        for gb in range(3):
            for gc in range(3):
                sa = group_sizes[ga]
                sb = group_sizes[gb]
                sc = group_sizes[gc]
                
                if sa == 2 and sb == 2 and sc == 2:
                    # Use Strassen rank-7
                    for (a_entries, b_entries, c_entries) in strassen_2x2:
                        u_vec = build_U_block(a_entries, ga, gb)
                        v_vec = build_V_block(b_entries, gb, gc)
                        w_vec = build_W_block(c_entries, gc, ga)
                        terms.append((u_vec, v_vec, w_vec))
                elif sa == 2 and sb == 2 and sc == 1:
                    # 2x2x1: rank 4
                    for li in range(2):
                        for lj in range(2):
                            u_vec = build_U_block([[(li, lj, 1)]], ga, gb)
                            v_vec = build_V_block([[(lj, 0, 1)]], gb, gc)
                            w_vec = build_W_block([[(0, li, 1)]], gc, ga)
                            terms.append((u_vec, v_vec, w_vec))
                elif sa == 2 and sb == 1 and sc == 2:
                    # 2x1x2: rank 4
                    for li in range(2):
                        for lk in range(2):
                            u_vec = build_U_block([[(li, 0, 1)]], ga, gb)
                            v_vec = build_V_block([[(0, lk, 1)]], gb, gc)
                            w_vec = build_W_block([[(lk, li, 1)]], gc, ga)
                            terms.append((u_vec, v_vec, w_vec))
                elif sa == 1 and sb == 2 and sc == 2:
                    # 1x2x2: rank 4
                    for lj in range(2):
                        for lk in range(2):
                            u_vec = build_U_block([[(0, lj, 1)]], ga, gb)
                            v_vec = build_V_block([[(lj, lk, 1)]], gb, gc)
                            w_vec = build_W_block([[(lk, 0, 1)]], gc, ga)
                            terms.append((u_vec, v_vec, w_vec))
                elif sa == 2 and sb == 1 and sc == 1:
                    # 2x1x1: rank 2
                    for li in range(2):
                        u_vec = build_U_block([[(li, 0, 1)]], ga, gb)
                        v_vec = build_V_block([[(0, 0, 1)]], gb, gc)
                        w_vec = build_W_block([[(0, li, 1)]], gc, ga)
                        terms.append((u_vec, v_vec, w_vec))
                elif sa == 1 and sb == 2 and sc == 1:
                    # 1x2x1: rank 2
                    for lj in range(2):
                        u_vec = build_U_block([[(0, lj, 1)]], ga, gb)
                        v_vec = build_V_block([[(lj, 0, 1)]], gb, gc)
                        w_vec = build_W_block([[(0, 0, 1)]], gc, ga)
                        terms.append((u_vec, v_vec, w_vec))
                elif sa == 1 and sb == 1 and sc == 2:
                    # 1x1x2: rank 2
                    for lk in range(2):
                        u_vec = build_U_block([[(0, 0, 1)]], ga, gb)
                        v_vec = build_V_block([[(0, lk, 1)]], gb, gc)
                        w_vec = build_W_block([[(lk, 0, 1)]], gc, ga)
                        terms.append((u_vec, v_vec, w_vec))
                elif sa == 1 and sb == 1 and sc == 1:
                    # 1x1x1: rank 1
                    u_vec = build_U_block([[(0, 0, 1)]], ga, gb)
                    v_vec = build_V_block([[(0, 0, 1)]], gb, gc)
                    w_vec = build_W_block([[(0, 0, 1)]], gc, ga)
                    terms.append((u_vec, v_vec, w_vec))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    for r, (u_vec, v_vec, w_vec) in enumerate(terms):
        for idx in range(25):
            U[idx][r] = u_vec[idx]
            V[idx][r] = v_vec[idx]
            W[idx][r] = w_vec[idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
