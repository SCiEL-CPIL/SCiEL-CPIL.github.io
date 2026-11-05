DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-196 tensor decomposition for 6x6 matrix multiplication
    using a two-level approach:
      Level 1: Strassen on 2x2 blocks of 3x3 (7 multiplications of 3x3)
      Level 2: For each 3x3 product, use a rank-28 decomposition
               (Strassen with padded 2x2 blocks: pad 3x3 to 4x4, use Strassen rank 7 * 4 = 28)
    
    Total rank = 7 * 28 = 196
    """
    import numpy as np
    
    n = m = p = 6
    rank = 196
    
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    # Level 1 Strassen terms: 7 terms, each computing a 3x3 * 3x3 product
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 3x3
    # 
    # T1: (A00 + A11) * (B00 + B11)
    # T2: (A10 + A11) * B00
    # T3: A00 * (B01 - B11)
    # T4: A11 * (B10 - B00)
    # T5: (A00 + A01) * B11
    # T6: (A10 - A00) * (B00 + B01)
    # T7: (A01 - A11) * (B10 + B11)
    #
    # Result:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # Each Strassen term t has a left factor X (3x3) and right factor Y (3x3)
    # that are linear combinations of the 3x3 blocks.
    # For the decomposition, each scalar product X[i,a] * Y[a,k] for i,a,k in 0..2
    # is one rank-1 term.
    #
    # For each term, we need to express X[i,a] and Y[a,k] as linear combos of
    # original A and B entries.
    #
    # Block (br, bc) covers A rows 3*br..3*br+2, cols 3*bc..3*bc+2
    # A entry at global (3*br+i, 3*bc+a)
    # B entry at global (3*br+a, 3*bc+k)  [since B's row index corresponds to column of A]
    
    # Define Strassen terms as (left_blocks, right_blocks)
    # where each is a list of (block_row, block_col, coefficient)
    strassen_terms = [
        # T1: (A00+A11)*(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # T2: (A10+A11)*B00
        ([(1,0,1),(1,1,1)], [(0,0,1)]),
        # T3: A00*(B01-B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),
        # T4: A11*(B10-B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),
        # T5: (A00+A01)*B11
        ([(0,0,1),(0,1,1)], [(1,1,1)]),
        # T6: (A10-A00)*(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # T7: (A01-A11)*(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
    ]
    
    # Result contributions: for each result block (rbr, rbc), list of (term_idx, sign)
    result_contribs = {
        (0,0): [(0,1),(3,1),(4,-1),(6,1)],
        (0,1): [(2,1),(4,1)],
        (1,0): [(1,1),(3,1)],
        (1,1): [(0,1),(1,-1),(2,1),(5,1)],
    }
    
    col = 0
    for t_idx, (left_blocks, right_blocks) in enumerate(strassen_terms):
        # For each (i, a, k) in 0..2, create one rank-1 term
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # X[i,a] = sum over (br,bc,coeff) in left_blocks: coeff * A[3*br+i, 3*bc+a]
                    # Y[a,k] = sum over (br,bc,coeff) in right_blocks: coeff * B[3*br+a, 3*bc+k]
                    
                    for (br, bc, coeff_l) in left_blocks:
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A, col] += coeff_l
                    
                    for (br, bc, coeff_r) in right_blocks:
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B, col] += coeff_r
                    
                    # W: this term contributes to result C[i_total, k_total]
                    # where i_total = 3*rbr + i, k_total = 3*rbc + k
                    # W row index = k_total * n + i_total (per trace convention W[k*n+i, r])
                    for (rbr, rbc) in result_contribs:
                        for (si, sign) in result_contribs[(rbr, rbc)]:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                W_row = k_total * 6 + i_total
                                W[W_row, col] += sign
                    
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
