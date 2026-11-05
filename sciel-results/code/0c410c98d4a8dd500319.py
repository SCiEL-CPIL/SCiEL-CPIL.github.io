"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    2-level recursive Strassen for 6x6 matrix multiplication.
    
    Level 1: Split 6x6 into 2x2 blocks of 3x3. Use Strassen: 7 terms.
    Level 2: Each 3x3 product is split into 2x2 blocks of sizes (2,1).
             Use Strassen: 7 terms per level-1 term.
    Base:    Each resulting product is of blocks with sizes in {1,2}.
             Schoolbook for each.
    
    Total rank = 7 * 7 * (scalar mults per level-2 term)
    
    For a level-2 Strassen term, the product is of two 3x3 block matrices
    where each 3x3 is composed of 2x2, 2x1, 1x2, 1x1 blocks.
    The 7 Strassen terms at level 2 produce 7 products, each of which
    is a matrix product of specific block shapes. We compute each via
    schoolbook scalar multiplications.
    
    Let me enumerate the 7 level-2 Strassen terms and count scalar mults:
    Blocks: A00(2x2), A01(2x1), A10(1x2), A11(1x1)
            B00(2x2), B01(2x1), B10(1x2), B11(1x1)
    
    M1 = (A00+A11)*(B00+B11): 
         Left is 3x3 (top-left 2x2 is A00, top-right 2x1 is A01... wait no)
         Actually A00+A11 doesn't make sense as blocks unless we pad.
         
    I need to be more careful. Strassen for rectangular blocks:
    The standard Strassen formula assumes square blocks of equal size.
    For unequal blocks, we can still apply the same algebraic identity
    but the "blocks" in the sums have different shapes.
    
    Let me define:
    A = [[A_tl, A_tr],     B = [[B_tl, B_tr],
         [A_bl, A_br]]          [B_bl, B_br]]
    where A_tl, B_tl are 2x2; A_tr, B_tr are 2x1;
          A_bl, B_bl are 1x2; A_br, B_br are 1x1.
    
    Strassen terms (valid for any block sizes as long as products are defined):
    M1 = (A_tl + A_br) * (B_tl + B_br)  -- 3x3 * 3x3? No!
         A_tl+A_br: A_tl is 2x2, A_br is 1x1. Can't add directly.
    
    This is the problem. Strassen requires same-sized blocks for addition.
    
    So I CANNOT directly apply Strassen to 3x3 with (2,1) split.
    
    Alternative: Pad 3x3 to 4x4, apply Strassen on 4x4 (2x2 blocks of 2x2),
    get rank 7*4=28 for 4x4, then the 3x3 result is embedded. Rank 28 > 27. Worse.
    
    So the best I can do is rank 189 with 1-level Strassen + schoolbook 3x3.
    
    Let me just verify the current code is correct and keep it, but clean it up.
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # 7 Strassen combinations at level 1 for 6x6 with 3x3 blocks
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),   # T1
        ({(1,0):1, (1,1):1}, {(0,0):1}),             # T2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),            # T3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),            # T4
        ({(0,0):1, (0,1):1}, {(1,1):1}),             # T5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),   # T6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),   # T7
    ]
    
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    for (rbr, rbc), signs in result_contribs.items():
                        for (si, sign) in signs:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                W_row = k_total * 6 + i_total
                                W[W_row][col] += sign
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
