"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-84 exact tensor decomposition for 4x4 by 4x6 matrix multiplication
    using Strassen's block decomposition.
    
    Strategy:
    - Partition A (4x4) into four 2x2 blocks.
    - Partition B (4x6) into four 2x3 blocks.
    - Apply Strassen's 7-step algorithm to each of the 4 block-pairs (i-block, j-block).
    - For each Strassen step, we need to handle the 3 columns of the 2x3 sub-blocks.
    - Total rank = 4 (block pairs) * 7 (Strassen steps) * 3 (columns) = 84.
    
    The output W follows the convention W[k*n+i, r] where k is column index of B (0..p-1),
    i is row index of A (0..n-1), n=4, p=6.
    """
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    # Strassen's 7 steps for 2x2 * 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4 * 4x6, we partition into 2x2 blocks of A and 2x3 blocks of B.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x3)
    # C blocks: C00, C01, C10, C11 (each 2x3)
    
    # For each block pair (Ai, Bj), we apply Strassen with 7 multiplications.
    # But since B blocks are 2x3, each "multiplication" involves 3 column-wise operations.
    # So each Strassen step for a block pair produces 3 rank-1 terms (one per column).
    
    # Total: 4 block pairs * 7 Strassen steps * 3 columns = 84 rank terms.

    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Helper functions to get flat indices
    def u_idx(i, j):
        """Flat index in U for element A[i,j]"""
        return i * m + j

    def v_idx(j, k):
        """Flat index in V for element B[j,k]"""
        return j * p + k

    def w_idx(i, k):
        """Flat index in W for element C[i,k], using convention W[k*n+i]"""
        return k * n + i

    # Block offset definitions:
    # A block (bi, bj) where bi, bj in {0,1}:
    #   bi=0: rows 0,1; bi=1: rows 2,3
    #   bj=0: cols 0,1; bj=1: cols 2,3
    # B block (bj, bk) where bj in {0,1}, bk in {0,1,2} (column groups):
    #   bj=0: rows 0,1; bj=1: rows 2,3
    #   bk=0: cols 0,1; bk=1: cols 2,3; bk=2: cols 4,5
    # Wait, B is 4x6, partitioned into 2x3 blocks:
    #   B00: rows 0-1, cols 0-2
    #   B01: rows 0-1, cols 3-5
    #   B10: rows 2-3, cols 0-2
    #   B11: rows 2-3, cols 3-5
    # So B has 2x2 block structure where each block is 2x3.
    
    # Let me redefine:
    # A is 4x4, partitioned into 2x2 blocks of size 2x2:
    #   A00: rows 0-1, cols 0-1
    #   A01: rows 0-1, cols 2-3
    #   A10: rows 2-3, cols 0-1
    #   A11: rows 2-3, cols 2-3
    #
    # B is 4x6, partitioned into 2x2 blocks of size 2x3:
    #   B00: rows 0-1, cols 0-2
    #   B01: rows 0-1, cols 3-5
    #   B10: rows 2-3, cols 0-2
    #   B11: rows 2-3, cols 3-5
    #
    # C is 4x6, partitioned into 2x2 blocks of size 2x3:
    #   C00: rows 0-1, cols 0-2
    #   C01: rows 0-1, cols 3-5
    #   C10: rows 2-3, cols 0-2
    #   C11: rows 2-3, cols 3-5

    # For Strassen on block pair (Ai, Bj) where Ai is 2x2 and Bj is 2x3:
    # We treat Bj as having 3 columns, and apply Strassen column by column.
    # Actually, Strassen operates on 2x2 matrices. For 2x2 * 2x3, we can
    # view it as 3 independent 2x2 * 2x1 multiplications, or apply the
    # Strassen formulas where the "B" matrices are 2x3 and the "M" results are 2x3.
    
    # The Strassen formulas work the same way:
    # M1 = (A11 + A22) * B11  where B11 is 2x3
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Here B11, B12, B21, B22 are 2x3 sub-blocks of the 2x3 B block.
    # Wait, that doesn't make sense. Let me reconsider.
    
    # Actually, for 2x2 * 2x3:
    # A = [[A11, A12], [A21, A22]] where each Ai is 1x1 (scalar)
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bj is 1x1 (scalar)
    # C = [[C11, C12, C13], [C21, C22, C23]]
    
    # Strassen for 2x2 * 2x2 (first two columns of B):
    # M1 = (A11+A22)(B11+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For the third column, we can use schoolbook (3 terms) or extend Strassen.
    # It's simpler to just use 7 Strassen terms for the first 2 columns and 3 schoolbook for the 3rd.
    # That gives 7 + 3 = 10 terms per block pair, total 4*10 = 40. That's better than 84!
    
    # Wait, but the inspiration program used rank 84. Let me check their approach.
    # They used 7 * 2 * 2 * 3 = 84. So they applied Strassen's 7 steps, and for each step,
    # they had 2 (i-block) * 2 (j-block) * 3 (k-sub) terms.
    
    # Actually, I think the correct interpretation is:
    # For each of the 4 block pairs (A block, B block), apply Strassen.
    # Strassen has 7 steps. For each step, the "multiplication" is between a 2x2 linear combo
    # and a 2x3 linear combo. This produces a 2x3 result, which requires 6 scalar multiplications
    # in schoolbook, but Strassen reduces the 2x2 part to 7 multiplications total for the block.
    
    # I'm getting confused. Let me just follow the inspiration program's approach which scored 1.1429.
    # They used rank = 7 * 2 * 2 * 3 = 84.
    
    # Let me re-read the inspiration code more carefully.
    # They define steps as tuples of (alpha, beta, gamma) where each is a dict of block->coeff.
    # Then for each step, for each i1 in range(2), j1 in range(2), k1 in range(3):
    #   they add one column to U, V, W.
    
    # So the 3 in their formula is for the 3 columns of the 2x3 B block.
    # The 2*2 is for the 2x2 block structure of A (and correspondingly B and C).
    
    # Let me just implement it the same way as the inspiration, which is known to work.

    # Redefine rank
    rank = 7 * 2 * 2 * 3  # 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen steps for 2x2 blocks
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # Blocks are labeled '00', '01', '10', '11'
    # '00' = top-left, '01' = top-right, '10' = bottom-left, '11' = bottom-right
    
    steps = [
        # M1 = (A00 + A11)(B00 + B11)
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        # M2 = (A10 + A11)B00
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        # M3 = A00(B01 - B11)
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        # M4 = A11(B10 - B00)
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        # M5 = (A00 + A01)B11
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    # Block index to flat index helpers
    # For A (4x4), block 'ij' where i in {0,1}, j in {0,1}:
    #   '00': rows 0-1, cols 0-1
    #   '01': rows 0-1, cols 2-3
    #   '10': rows 2-3, cols 0-1
    #   '11': rows 2-3, cols 2-3
    
    def get_u_row(block, i1, j1):
        """Get flat index in U for block 'block' with local row i1, col j1 within the 2x2 block"""
        if block == '00':
            i, j = i1, j1
        elif block == '01':
            i, j = i1, j1 + 2
        elif block == '10':
            i, j = i1 + 2, j1
        elif block == '11':
            i, j = i1 + 2, j1 + 2
        else:
            raise ValueError(f"Invalid block {block}")
        return i * m + j

    # For B (4x6), block 'ij' where i in {0,1}, j in {0,1}:
    #   '00': rows 0-1, cols 0-2
    #   '01': rows 0-1, cols 3-5
    #   '10': rows 2-3, cols 0-2
    #   '11': rows 2-3, cols 3-5
    
    def get_v_row(block, j1, k1):
        """Get flat index in V for block 'block' with local row j1, col k1 within the 2x3 block"""
        if block == '00':
            j, k = j1, k1
        elif block == '01':
            j, k = j1, k1 + 3
        elif block == '10':
            j, k = j1 + 2, k1
        elif block == '11':
            j, k = j1 + 2, k1 + 3
        else:
            raise ValueError(f"Invalid block {block}")
        return j * p + k

    # For C (4x6), block 'ij' where i in {0,1}, j in {0,1}:
    #   '00': rows 0-1, cols 0-2
    #   '01': rows 0-1, cols 3-5
    #   '10': rows 2-3, cols 0-2
    #   '11': rows 2-3, cols 3-5
    
    def get_w_row(block, i1, k1):
        """Get flat index in W for block 'block' with local row i1, col k1 within the 2x3 block
           using convention W[k*n+i]"""
        if block == '00':
            i, k = i1, k1
        elif block == '01':
            i, k = i1, k1 + 3
        elif block == '10':
            i, k = i1 + 2, k1
        elif block == '11':
            i, k = i1 + 2, k1 + 3
        else:
            raise ValueError(f"Invalid block {block}")
        return k * n + i

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        for i1 in range(2):
            for j1 in range(2):
                for k1 in range(3):
                    for b, val in alpha.items():
                        U[get_u_row(b, i1, j1), col] += val
                    for b, val in beta.items():
                        V[get_v_row(b, j1, k1), col] += val
                    for b, val in gamma.items():
                        W[get_w_row(b, i1, k1), col] += val
                    col += 1

    # Deadline guard
    if time.time() - start > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
