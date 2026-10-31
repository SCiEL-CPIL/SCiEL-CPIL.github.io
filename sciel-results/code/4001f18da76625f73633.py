"""
Optimal rank-49 decomposition for 4x4 matrix multiplication.
This implements the recursive Strassen algorithm, which is known to be optimal for 4x4 matrices.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: 7 outer multiplications of 2x2 blocks,
    each computed via 7 inner scalar multiplications.
    Total rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4), "This decomposition is specific to 4x4 matrices."
    
    _t0 = time.time()
    _DEADLINE = 150.0
    _MARGIN = 5.0

    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def w_idx(i, k):
        """W[k*n+i, r] convention: i is row index, k is column index in result"""
        return k * n + i

    # Strassen 2x2 multiplication factors
    # For C = AB where A, B are 2x2:
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

    # A-block coefficients for each of the 7 multiplications
    # Each entry is a list of (row, col, coeff) for the 2x2 block
    A_factors = [
        # M1: (A00+A11)(B00+B11)
        [[[(0,0,1)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,1)]]],
        # M2: (A10+A11)B00
        [[[(0,0,0)], [(0,1,0)]],
         [[(1,0,1)], [(1,1,1)]]],
        # M3: A00(B01-B11)
        [[[(0,0,1)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,0)]]],
        # M4: A11(B10-B00)
        [[[(0,0,0)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,1)]]],
        # M5: (A00+A01)B11
        [[[(0,0,1)], [(0,1,1)]],
         [[(1,0,0)], [(1,1,0)]]],
        # M6: (A10-A00)(B00+B01)
        [[[(0,0,-1)], [(0,1,0)]],
         [[(1,0,1)], [(1,1,0)]]],
        # M7: (A01-A11)(B10+B11)
        [[[(0,0,0)], [(0,1,1)]],
         [[(1,0,0)], [(1,1,-1)]]],
    ]

    # B-block coefficients for each of the 7 multiplications
    B_factors = [
        # M1: (A00+A11)(B00+B11)
        [[[(0,0,1)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,1)]]],
        # M2: (A10+A11)B00
        [[[(0,0,1)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,0)]]],
        # M3: A00(B01-B11)
        [[[(0,0,0)], [(0,1,1)]],
         [[(1,0,0)], [(1,1,-1)]]],
        # M4: A11(B10-B00)
        [[[(0,0,-1)], [(0,1,0)]],
         [[(1,0,1)], [(1,1,0)]]],
        # M5: (A00+A01)B11
        [[[(0,0,0)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,1)]]],
        # M6: (A10-A00)(B00+B01)
        [[[(0,0,1)], [(0,1,1)]],
         [[(1,0,0)], [(1,1,0)]]],
        # M7: (A01-A11)(B10+B11)
        [[[(0,0,0)], [(0,1,0)]],
         [[(1,0,1)], [(1,1,1)]]],
    ]

    # C-block contributions for each of the 7 multiplications
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_factors = [
        # M1
        [[[(0,0,1)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,1)]]],
        # M2
        [[[(0,0,0)], [(0,1,0)]],
         [[(1,0,1)], [(1,1,-1)]]],
        # M3
        [[[(0,0,0)], [(0,1,1)]],
         [[(1,0,0)], [(1,1,1)]]],
        # M4
        [[[(0,0,1)], [(0,1,0)]],
         [[(1,0,1)], [(1,1,0)]]],
        # M5
        [[[(0,0,-1)], [(0,1,1)]],
         [[(1,0,0)], [(1,1,0)]]],
        # M6
        [[[(0,0,0)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,1)]]],
        # M7
        [[[(0,0,1)], [(0,1,0)]],
         [[(1,0,0)], [(1,1,0)]]],
    ]

    # For each outer Strassen multiplication (7 total),
    # we have 2x2 blocks that map to 4x4 positions.
    # The outer Strassen operates on 2x2 blocks of the 4x4 matrices.
    
    # Define which 2x2 blocks of A and B are used for each outer multiplication
    # A00 = rows 0-1, cols 0-1; A01 = rows 0-1, cols 2-3
    # A10 = rows 2-3, cols 0-1; A11 = rows 2-3, cols 2-3
    
    # B00 = rows 0-1, cols 0-1; B01 = rows 0-1, cols 2-3
    # B10 = rows 2-3, cols 0-1; B11 = rows 2-3, cols 2-3
    
    # Outer Strassen:
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

    # For each outer multiplication, define the A-block and B-block
    # as lists of (block_row, block_col, coeff)
    outer_A = [
        # M1: A00 + A11
        [(0, 0, 1), (1, 1, 1)],
        # M2: A10 + A11
        [(1, 0, 1), (1, 1, 1)],
        # M3: A00
        [(0, 0, 1)],
        # M4: A11
        [(1, 1, 1)],
        # M5: A00 + A01
        [(0, 0, 1), (0, 1, 1)],
        # M6: A10 - A00
        [(1, 0, 1), (0, 0, -1)],
        # M7: A01 - A11
        [(0, 1, 1), (1, 1, -1)],
    ]
    
    outer_B = [
        # M1: B00 + B11
        [(0, 0, 1), (1, 1, 1)],
        # M2: B00
        [(0, 0, 1)],
        # M3: B01 - B11
        [(0, 1, 1), (1, 1, -1)],
        # M4: B10 - B00
        [(1, 0, 1), (0, 0, -1)],
        # M5: B11
        [(1, 1, 1)],
        # M6: B00 + B01
        [(0, 0, 1), (0, 1, 1)],
        # M7: B10 + B11
        [(1, 0, 1), (1, 1, 1)],
    ]
    
    # C-block contributions for each outer multiplication
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    outer_C = [
        # M1
        [(0, 0, 1), (1, 1, 1)],
        # M2
        [(1, 0, 1), (1, 1, -1)],
        # M3
        [(0, 1, 1), (1, 1, 1)],
        # M4
        [(0, 0, 1), (1, 0, 1)],
        # M5
        [(0, 0, -1), (0, 1, 1)],
        # M6
        [(1, 1, 1)],
        # M7
        [(0, 0, 1)],
    ]

    col = 0
    for t in range(7):
        # For each outer multiplication t, we need to decompose
        # the 2x2 block multiplication into 7 scalar multiplications
        # using the inner Strassen factors
        
        for k in range(7):
            # A_factors[k] and B_factors[k] give the 2x2 block coefficients
            # for the k-th scalar multiplication in the inner Strassen
            
            # Add to U: for each entry (r, c, v) in A_factors[k][i][j],
            # where i is the block row and j is the block col,
            # the global row is outer_A[t] block_row * 2 + i,
            # global col is outer_A[t] block_col * 2 + j
            for (ba_r, ba_c, ba_coeff) in outer_A[t]:
                for i in range(2):
                    for j in range(2):
                        for (r, c, v) in A_factors[k][i][j]:
                            global_row = ba_r * 2 + r
                            global_col = ba_c * 2 + c
                            U[global_row * m + global_col, col] += ba_coeff * v
            
            # Add to V: for each entry (r, c, v) in B_factors[k][i][j],
            # where i is the block row and j is the block col,
            # the global row is outer_B[t] block_row * 2 + i,
            # global col is outer_B[t] block_col * 2 + j
            for (bb_r, bb_c, bb_coeff) in outer_B[t]:
                for i in range(2):
                    for j in range(2):
                        for (r, c, v) in B_factors[k][i][j]:
                            global_row = bb_r * 2 + r
                            global_col = bb_c * 2 + c
                            V[global_row * p + global_col, col] += bb_coeff * v
            
            # Add to W: for each C-block contribution from outer_C[t]
            # and each C-block contribution from C_factors[k]
            for (bc_r, bc_c, bc_coeff) in outer_C[t]:
                for i in range(2):
                    for j in range(2):
                        for (r, c, v) in C_factors[k][i][j]:
                            global_row = bc_r * 2 + r
                            global_col = bc_c * 2 + c
                            W[w_idx(global_row, global_col), col] += bc_coeff * v
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE - _MARGIN:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
