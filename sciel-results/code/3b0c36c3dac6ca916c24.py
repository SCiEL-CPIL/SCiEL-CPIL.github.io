"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Use block decomposition:
    # Split A (2x4) into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # Split B (4x5) into B_top (2x5, rows 0-1) and B_bottom (2x5, rows 2-3)
    # C = A_left * B_top + A_right * B_bottom
    # Each sub-product is 2x2x5.
    # For each 2x2x5: split B's 5 cols into 2+2+1
    # Use Strassen (rank 7) for 2x2x2, schoolbook (rank 4) for 2x2x1
    # Total per 2x2x5: 7+7+4 = 18
    # Total rank: 18+18 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    def add_strassen_2x2x2(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        """Add Strassen decomposition for 2x2 * 2x2 block.
        a_i0, a_j0: top-left corner of A block (row, col)
        b_i0, b_j0: top-left corner of B block (row, col)
        c_i0, c_k0: top-left corner of C block (row i, col k)
        """
        def a_idx(li, lj):
            return (a_i0 + li) * m + (a_j0 + lj)
        def b_idx(bi, bj):
            return (b_i0 + bi) * p + (b_j0 + bj)
        def c_idx(ci, ck):
            # W index: k*n + i
            return (c_k0 + ck) * n + (c_i0 + ci)
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        u = {a_idx(0,0): 1, a_idx(1,1): 1}
        v = {b_idx(0,0): 1, b_idx(1,1): 1}
        w = {c_idx(0,0): 1, c_idx(1,1): 1}
        add_rank1_term(u, v, w)
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        u = {a_idx(1,0): 1, a_idx(1,1): 1}
        v = {b_idx(0,0): 1}
        w = {c_idx(1,0): 1, c_idx(1,1): -1}
        add_rank1_term(u, v, w)
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        u = {a_idx(0,0): 1}
        v = {b_idx(0,1): 1, b_idx(1,1): -1}
        w = {c_idx(0,1): 1, c_idx(1,1): 1}
        add_rank1_term(u, v, w)
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        u = {a_idx(1,1): 1}
        v = {b_idx(1,0): 1, b_idx(0,0): -1}
        w = {c_idx(0,0): 1, c_idx(1,0): 1}
        add_rank1_term(u, v, w)
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        u = {a_idx(0,0): 1, a_idx(0,1): 1}
        v = {b_idx(1,1): 1}
        w = {c_idx(0,0): -1, c_idx(0,1): 1}
        add_rank1_term(u, v, w)
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        u = {a_idx(1,0): 1, a_idx(0,0): -1}
        v = {b_idx(0,0): 1, b_idx(0,1): 1}
        w = {c_idx(1,1): 1}
        add_rank1_term(u, v, w)
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        u = {a_idx(0,1): 1, a_idx(1,1): -1}
        v = {b_idx(1,0): 1, b_idx(1,1): 1}
        w = {c_idx(0,0): 1}
        add_rank1_term(u, v, w)
    
    def add_schoolbook_2x2x1(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        """Add schoolbook decomposition for 2x2 * 2x1 block.
        A is 2x2, B is 2x1 (single column), C is 2x1.
        """
        def a_idx(li, lj):
            return (a_i0 + li) * m + (a_j0 + lj)
        def b_idx(bi):
            return (b_i0 + bi) * p + b_j0
        def c_idx(ci):
            return c_k0 * n + (c_i0 + ci)
        
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        for ci in range(2):
            for aj in range(2):
                u = {a_idx(ci, aj): 1}
                v = {b_idx(aj): 1}
                w = {c_idx(ci): 1}
                add_rank1_term(u, v, w)
    
    def add_2x2x5(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        """Add decomposition for 2x2 * 2x5.
        Split B's 5 columns into 2+2+1.
        """
        # First 2 columns of B
        add_strassen_2x2x2(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0)
        # Next 2 columns of B
        add_strassen_2x2x2(a_i0, a_j0, b_i0, b_j0 + 2, c_i0, c_k0 + 2)
        # Last 1 column of B
        add_schoolbook_2x2x1(a_i0, a_j0, b_i0, b_j0 + 4, c_i0, c_k0 + 4)
    
    # Block 1: A_left (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-4)
    add_2x2x5(0, 0, 0, 0, 0, 0)
    
    # Block 2: A_right (rows 0-1, cols 2-3) * B_bottom (rows 2-3, cols 0-4)
    add_2x2x5(0, 2, 2, 0, 0, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
