"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen's algorithm on 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = np.zeros((8, rank))
    V = np.zeros((24, rank))
    W = np.zeros((12, rank))
    
    # Strassen terms for 2x2*2x2
    # Indices: P (2x2), Q (2x2), R (2x2)
    # U factors: (row, col) in P
    # V factors: (row, col) in Q
    # W factors: (row, col) in R
    
    strassen_terms = [
        # M1 = (P00+P11)*(Q00+Q11)
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((0,1), 0), ((1,0), 0), ((1,1), 1)]),
        # M2 = (P10+P11)*Q00
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # M3 = P00*(Q01-Q11)
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # M4 = P11*(Q10-Q00)
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # M5 = (P00+P01)*Q11
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # M6 = (P10-P00)*(Q00+Q01)
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # M7 = (P01-P11)*(Q10+Q11)
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    col_idx = 0
    for source in range(2):  # 0: A rows 0-1, B rows 0-1; 1: A rows 2-3, B rows 2-3
        for c in range(3):  # col blocks of B (and C)
            for t in range(7):
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                for (pi, pj), coeff in u_factors:
                    # A is 2x4. source 0 uses cols 0,1. source 1 uses cols 2,3.
                    # pi is 0 or 1 (row in 2x2 block).
                    # u_row = pi * 4 + (pj + 2*source)
                    u_row = pi * 4 + (pj + 2*source)
                    U[u_row, col_idx] += coeff
                
                for (qi, qj), coeff in v_factors:
                    # B is 4x6. source 0 uses rows 0,1. source 1 uses rows 2,3.
                    # c is col block 0,1,2. qj is 0 or 1.
                    # v_row = (qi + 2*source) * 6 + (2*c + qj)
                    v_row = (qi + 2*source) * 6 + (2*c + qj)
                    V[v_row, col_idx] += coeff
                
                for (ri, rj), coeff in w_factors:
                    # C is 2x6.
                    # ri is 0 or 1 (row). rj is 0 or 1 (col in 2x2 block).
                    # c is col block.
                    # w_row = (2*c + rj) * 2 + ri
                    w_row = (2*c + rj) * 2 + ri
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END
