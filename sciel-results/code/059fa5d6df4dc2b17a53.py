"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    This is constructed by applying Strassen's algorithm (rank 7) to four 2x2x2 sub-problems.
    The 2x4 matrix A is split into two 2x2 blocks horizontally.
    The 4x4 matrix B is split into four 2x2 blocks.
    The 2x4 matrix C is split into two 2x2 blocks horizontally.
    
    C_left = A_left * B_top_left + A_right * B_bottom_left
    C_right = A_left * B_top_right + A_right * B_bottom_right
    
    Each term is a 2x2 * 2x2 multiplication, solvable with rank 7.
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 2x2x2 decomposition indices:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Coefficients for U (A-side), V (B-side), W (C-side) for each Strassen product s in 0..6
    # U row index: i * m + j. Here i is local row (0,1), j is local col (0,1) within the 2x2 block.
    # V row index: j * p + k. Here j is local row (0,1), k is local col (0,1).
    # W row index: k * n + i. Here k is local col (0,1), i is local row (0,1).
    
    # U_contrib[s] is a list of (local_i, local_j, coeff)
    # V_contrib[s] is a list of (local_j, local_k, coeff)
    # W_contrib[s] is a list of (local_k, local_i, coeff)
    
    U_coeffs = [
        [(0, 0, 1), (1, 1, 1)],       # m1: a11 + a22
        [(1, 0, 1), (1, 1, 1)],       # m2: a21 + a22
        [(0, 0, 1)],                  # m3: a11
        [(1, 1, 1)],                  # m4: a22
        [(0, 0, 1), (0, 1, 1)],       # m5: a11 + a12
        [(1, 0, 1), (0, 0, -1)],      # m6: a21 - a11
        [(0, 1, 1), (1, 1, -1)]       # m7: a12 - a22
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],       # m1: b11 + b22
        [(0, 0, 1)],                  # m2: b11
        [(0, 1, 1), (1, 1, -1)],      # m3: b12 - b22
        [(1, 0, 1), (0, 0, -1)],      # m4: b21 - b11
        [(1, 1, 1)],                  # m5: b22
        [(0, 0, 1), (0, 1, 1)],       # m6: b11 + b12
        [(1, 0, 1), (1, 1, 1)]        # m7: b21 + b22
    ]
    
    W_coeffs = [
        [(0, 0, 1), (0, 1, -1), (1, 0, 1), (1, 1, 1)],  # m1: c11(1) + c22(1) ... wait
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # Let's map W contributions carefully.
        # W[k*n+i] where k is col index (0,1), i is row index (0,1).
        # c11 is W[0*2+0] = W[0]
        # c12 is W[1*2+0] = W[2]
        # c21 is W[0*2+1] = W[1]
        # c22 is W[1*2+1] = W[3]
        
        # m1: c11 += 1, c22 += 1
        [(0, 0, 1), (1, 1, 1)],
        # m2: c21 += 1, c22 -= 1
        [(0, 1, 1), (1, 1, -1)],
        # m3: c12 += 1, c22 += 1
        [(1, 0, 1), (1, 1, 1)],
        # m4: c11 += 1, c21 += 1
        [(0, 0, 1), (0, 1, 1)],
        # m5: c11 -= 1, c12 += 1
        [(0, 0, -1), (1, 0, 1)],
        # m6: c22 += 1
        [(1, 1, 1)],
        # m7: c11 += 1
        [(0, 0, 1)]
    ]

    col = 0
    for b_block in range(2):  # B column block: 0 (cols 0,1) or 1 (cols 2,3)
        for a_block in range(2):  # A column block / B row block: 0 (cols/rows 0,1) or 1 (cols/rows 2,3)
            for s in range(7):
                c = col
                
                # Global indices
                # A: rows 0,1. Cols: 2*a_block, 2*a_block+1
                # B: Rows: 2*a_block, 2*a_block+1. Cols: 2*b_block, 2*b_block+1
                # C: Rows: 0,1. Cols: 2*b_block, 2*b_block+1
                
                for (li, lj, coeff) in U_coeffs[s]:
                    # Global i = li (0 or 1)
                    # Global j = 2*a_block + lj
                    U[li * 4 + (2 * a_block + lj)][c] += coeff
                
                for (lj, lk, coeff) in V_coeffs[s]:
                    # Global j = 2*a_block + lj
                    # Global k = 2*b_block + lk
                    V[(2 * a_block + lj) * 4 + (2 * b_block + lk)][c] += coeff
                
                for (lk, li, coeff) in W_coeffs[s]:
                    # Global k = 2*b_block + lk
                    # Global i = li (0 or 1)
                    W[(2 * b_block + lk) * 2 + li][c] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
