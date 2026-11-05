"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    
    # We use the Strassen algorithm for 2x2 blocks of size 3x3.
    # This reduces the problem to 7 multiplications of 3x3 matrices.
    # Each 3x3 multiplication is done with schoolbook rank 27.
    # Total rank = 7 * 27 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a schoolbook decomposition of a 3x3 multiplication
    # into the global U, V, W matrices at a specific offset.
    # The 3x3 blocks are indexed by (bi, bj) where bi, bj in {0, 1}.
    # Block (bi, bj) corresponds to rows/cols [3*bi : 3*bi+3].
    
    def add_3x3_mult(U, V, W, rank_offset, block_A, block_B, block_C):
        # block_A, block_B, block_C are tuples (i_offset, j_offset) for 3x3 blocks
        # Actually, we need to map the logical 3x3 indices to global 6x6 indices.
        # Let's define the 3x3 blocks explicitly.
        # A is 6x6. Blocks A00, A01, A10, A11.
        # A00 is rows 0-2, cols 0-2.
        # A01 is rows 0-2, cols 3-5.
        # A10 is rows 3-5, cols 0-2.
        # A11 is rows 3-5, cols 3-5.
        
        # We are adding the decomposition of C_block = A_block * B_block.
        # The global indices for the 3x3 multiplication:
        # A_block is at (r_a, c_a) in the 2x2 block grid.
        # B_block is at (r_b, c_b) in the 2x2 block grid.
        # C_block is at (r_c, c_c) in the 2x2 block grid.
        
        # We iterate over the 27 terms of the 3x3 schoolbook multiplication.
        # Local indices i, j, k in 0..2
        # Global row in A: 3 * r_a + i
        # Global col in A: 3 * c_a + j
        # Global row in B: 3 * r_b + j
        # Global col in B: 3 * c_b + k
        # Global row in C: 3 * r_c + k
        # Global col in C: 3 * c_c + i
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col_idx = rank_offset + i * 9 + j * 3 + k
                    # U index: row in A, col in "multiplication space"
                    # U is (n*m, rank) -> (36, 189)
                    # A is 6x6.
                    # U[global_row_A, col_idx] = 1
                    row_A = 3 * block_A[0] + i
                    col_A = 3 * block_A[1] + j
                    U[row_A * m + col_A][col_idx] = 1
                    
                    # V is (m*p, rank) -> (36, 189)
                    # B is 6x6.
                    row_B = 3 * block_B[0] + j
                    col_B = 3 * block_B[1] + k
                    V[row_B * p + col_B][col_idx] = 1
                    
                    # W is (p*n, rank) -> (36, 189)
                    # C is 6x6.
                    # W[k*n + i, r] convention?
                    # Task says: W[k*n+i, r].
                    # C[k, i] = sum A[i, j] B[j, k].
                    # So W index is (k, i) flattened?
                    # "W[k*n+i, r]"
                    # So row index in W is k*n + i.
                    row_W = 3 * block_C[0] + k
                    col_W = 3 * block_C[1] + i
                    W[row_W * n + col_W][col_idx] = 1

    # Strassen formulas for 2x2 blocks:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to expand these into linear combinations of 3x3 multiplications.
    # Each term like (A00 + A11) is a sum of two blocks.
    # The product (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This requires 4 separate 3x3 multiplications if we don't combine them.
    # Wait, Strassen's trick is that the *number of multiplications* is reduced.
    # But the multiplications are of *sums* of blocks.
    # To decompose this into the tensor form, we must treat the "sums" as new inputs.
    # However, the tensor decomposition must be linear in the original entries.
    # So we cannot just say "multiply (A00+A11) by (B00+B11)" as a single rank-27 block.
    # We must expand it into 4 rank-27 blocks?
    # If we expand M1 into 4 blocks, we use 4 * 27 = 108 rank for M1.
    # Then total rank = 7 * 108? No.
    
    # Let's re-evaluate.
    # The rank of the 2x2 Strassen algorithm is 7 * rank(M_3).
    # This is true if the "multiplications" are of the form X * Y where X and Y are linear combinations of blocks.
    # In the tensor decomposition, each "multiplication" X*Y where X, Y are 3x3 matrices
    # requires rank 27 (if schoolbook) or 23 (if optimal).
    # So if we use schoolbook for the 3x3 multiplications, we need 7 * 27 = 189 rank.
    # This is correct.
    # The expansion is:
    # M1 = (A00 + A11)(B00 + B11).
    # This is ONE multiplication of two 3x3 matrices (which happen to be sums of blocks).
    # So we just need to define the "input" to this multiplication as the sum of the corresponding blocks.
    # In the U, V, W matrices, we set the entries for the "summed" blocks.
    
    # So for M1:
    # U part: A00 and A11 contribute to the "A" side of M1.
    # V part: B00 and B11 contribute to the "B" side of M1.
    # W part: The result contributes to C00, C01, C10, C11 with signs.
    
    # Let's define the 7 multiplications.
    # Each multiplication has an "A-side" (linear combo of 3x3 blocks of A)
    # and a "B-side" (linear combo of 3x3 blocks of B).
    # And a "C-side" (linear combo of 3x3 blocks of C).
    
    # We will iterate over the 27 local indices for each of the 7 multiplications.
    
    # Multiplication 1: (A00 + A11)(B00 + B11)
    # A-side: A00 (+), A11 (+)
    # B-side: B00 (+), B11 (+)
    # C-side: C00 (+), C11 (+), C01 (-), C10 (-) ?
    # Let's check the C formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes:
    # to C00 with +1
    # to C11 with +1
    # M2 contributes:
    # to C10 with +1
    # to C11 with -1
    # M3 contributes:
    # to C01 with +1
    # to C11 with +1
    # M4 contributes:
    # to C00 with +1
    # to C10 with +1
    # M5 contributes:
    # to C00 with -1
    # to C01 with +1
    # M6 contributes:
    # to C11 with +1
    # M7 contributes:
    # to C00 with +1
    
    # We define the multiplications as:
    # (A_blocks_with_coeffs, B_blocks_with_coeffs, C_blocks_with_coeffs)
    
    mults = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10 + A11) B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]), # C10 +1, C11 -1
        # M3: A00 (B01 - B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]), # C01 +1, C11 +1
        # M4: A11 (B10 - B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]), # C00 +1, C10 +1
        # M5: (A00 + A01) B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]), # C00 -1, C01 +1
        # M6: (A10 - A00)(B00 + B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]), # C11 +1
        # M7: (A01 - A11)(B10 + B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]), # C00 +1
    ]
    
    # Wait, the C-side coefficients must match the Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify my C-side definitions above:
    # M1: C00 +1, C11 +1. (Correct: C00 has M1, C11 has M1)
    # M2: C10 +1, C11 -1. (Correct: C10 has M2, C11 has -M2)
    # M3: C01 +1, C11 +1. (Correct: C01 has M3, C11 has M3)
    # M4: C00 +1, C10 +1. (Correct: C00 has M4, C10 has M4)
    # M5: C00 -1, C01 +1. (Correct: C00 has -M5, C01 has M5)
    # M6: C11 +1. (Correct: C11 has M6)
    # M7: C00 +1. (Correct: C00 has M7)
    
    # The definitions look correct.
    
    # Now we fill U, V, W.
    # For each multiplication t in 0..6:
    #   offset = t * 27
    #   For each local i, j, k in 0..2:
    #     col = offset + i*9 + j*3 + k
    #     For each A-block (bi, bj, coeff) in A-side:
    #       row_A = 3*bi + i
    #       col_A = 3*bj + j
    #       U[row_A*m + col_A][col] += coeff
    #     For each B-block (bi, bj, coeff) in B-side:
    #       row_B = 3*bi + j
    #       col_B = 3*bj + k
    #       V[row_B*p + col_B][col] += coeff
    #     For each C-block (bi, bj, coeff) in C-side:
    #       row_C = 3*bi + k
    #       col_C = 3*bj + i
    #       W[row_C*n + col_C][col] += coeff
        
    for t in range(7):
        a_side, b_side, c_side = mults[t]
        offset = t * 27
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = offset + i * 9 + j * 3 + k
                    # A side
                    for bi, bj, coeff in a_side:
                        row_A = 3 * bi + i
                        col_A = 3 * bj + j
                        U[row_A * m + col_A][col] += coeff
                    # B side
                    for bi, bj, coeff in b_side:
                        row_B = 3 * bi + j
                        col_B = 3 * bj + k
                        V[row_B * p + col_B][col] += coeff
                    # C side
                    for bi, bj, coeff in c_side:
                        row_C = 3 * bi + k
                        col_C = 3 * bj + i
                        W[row_C * n + col_C][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
