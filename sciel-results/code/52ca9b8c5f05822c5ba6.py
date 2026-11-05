"""Improved program: 2-level recursive Strassen for 6x6x6 matrix multiplication.
Level 1: 6x6 -> 2x2 blocks of 3x3, rank 7
Level 2: each 3x3 -> 2x2 blocks of (2,1)x(2,1), rank 7
Total rank = 7 * 7 * 4 = 196... but actually the sub-blocks aren't all 2x2.

Let me use the correct recursive approach:
Level 1: 7 Strassen terms, each a 3x3 matrix multiply
Level 2: For each 3x3, use Strassen with blocks of size 2 and 1
  The 7 sub-products have varying sizes. Let me count carefully.

For 3x3 = [[A00(2x2), A01(2x1)], [A10(1x2), A11(1x1)]]:
  M1 = (A00+A11)(B00+B11): (2x2)(2x2) -> but A11 is 1x1, so A00+A11 needs padding
  Actually this doesn't work cleanly.

Let me use a different approach: 
Rank(6,6,6) <= 7 * Rank(3,3,3) <= 7 * 27 = 189 (current)
Or: Rank(6,6,6) <= 7 * 7 * 4 = 196 (worse)

Better: Use the fact that 6 = 2*3, and apply Strassen at both levels properly.
Actually, the cleanest improvement: use rank 189 but verify it's correct.

Let me try yet another approach: 
Partition 6x6 into 3x3 blocks (2x2 grid), use Strassen (7 terms).
For each 3x3 multiplication, use the optimal known decomposition.
The best known rank for 3x3 is 23 (Bini et al.), but that's hard to implement.
A simpler improvement: use Strassen on 3x3 with uneven split (2+1).

For 3x3 with blocks A = [[A11(2x2), A12(2x1)], [A21(1x2), A22(1x1)]]:
The Strassen-like decomposition needs careful handling.

Actually, let me just use the 189 rank solution but make it cleaner and verify.
The current code should work. Let me trace through it carefully.

Wait - I can do better. Instead of schoolbook for the 3x3 (rank 27),
I can use a rank-22 or rank-23 decomposition for 3x3.

Actually, the simplest valid improvement: 
Use 2-level Strassen where level 2 uses the (2,1) split properly.

For a 3x3 product C = A*B where:
A = [[A00, A01],     B = [[B00, B01],
     [A10, A11]]           [B10, B11]]
A00 is 2x2, A01 is 2x1, A10 is 1x2, A11 is 1x1
B00 is 2x2, B01 is 2x1, B10 is 1x2, B11 is 1x1

The 7 Strassen terms and their sub-product sizes:
M1 = (A00 + A11_padded)(B00 + B11_padded) where padding makes them 2x2
  A00 + pad(A11) is 2x2, B00 + pad(B11) is 2x2 -> 2x2 * 2x2 = 4 mults
M2 = (A10_padded + A11_padded) * B00: (1x2 pad to 2x2) * 2x2 = 4 mults
  Wait, A10 is 1x2, A11 is 1x1. A10 + A11 doesn't make sense dimensionally.
  
This is the problem with uneven Strassen. Let me just stick with rank 189
and make sure the code is correct.

Actually, I realize I can get rank 175 by using a different decomposition.
But let me first make sure 189 works, then try to improve.

Let me reconsider: the current program claims rank 189. Let me verify the logic.
7 Strassen terms at level 1, each decomposed into 27 schoolbook mults = 189.
This should be valid. Let me just clean up the code and ensure correctness.
"""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n = m = p = 6
    # Use 2-level decomposition:
    # Level 1: Strassen on 6x6 with 3x3 blocks -> 7 terms
    # Level 2: Schoolbook on each 3x3 -> 27 terms each
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen level 1: 6x6 split into 2x2 blocks of 3x3
    # A = [[A00, A01],   B = [[B00, B01],   C = [[C00, C01],
    #       [A10, A11]]       [B10, B11]]       [C10, C11]]
    #
    # T1 = (A00+A11)(B00+B11)
    # T2 = (A10+A11)B00
    # T3 = A00(B01-B11)
    # T4 = A11(B10-B00)
    # T5 = (A00+A01)B11
    # T6 = (A10-A00)(B00+B01)
    # T7 = (A01-A11)(B10+B11)
    #
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # Each Tj is a 3x3 matrix multiplication X*Y where X and Y are 3x3
    # linear combinations of the 3x3 blocks.
    
    # Define the 7 Strassen terms as (left_blocks, right_blocks)
    # Each block is identified by (row_block, col_block) where each is 0 or 1
    # Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    
    strassen_terms = [
        # (left: {(br,bc): coeff}, right: {(br,bc): coeff})
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # T1
        ({(1,0):1, (1,1):1}, {(0,0):1}),                 # T2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),                # T3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),                # T4
        ({(0,0):1, (0,1):1}, {(1,1):1}),                 # T5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),       # T6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),       # T7
    ]
    
    # Result contributions: which Strassen terms contribute to which C block with what sign
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        # For this Strassen term, compute X*Y where X and Y are 3x3 matrices
        # X[i,a] = sum_{(br,bc) in left_combo} coeff * A[3*br+i, 3*bc+a]
        # Y[a,k] = sum_{(br,bc) in right_combo} coeff * B[3*br+a, 3*bc+k]
        # Z[i,k] = sum_a X[i,a] * Y[a,k]
        #
        # Schoolbook decomposition: for each (i, a, k), one scalar mult
        # U: coeff of A[3*br+i, 3*bc+a] for each block in left_combo
        # V: coeff of B[3*br+a, 3*bc+k] for each block in right_combo
        # W: contributes to C[3*rbr+i, 3*rbc+k] for each result block
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # Fill U
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # Fill V
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # Fill W
                    for (rbr, rbc), signs in result_contribs.items():
                        for (si, sign) in signs:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                # W convention: W[k*n + i, r]
                                W_row = k_total * 6 + i_total
                                W[W_row][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
