"""Strassen-based rank-189 decomposition for 6x6 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # 2x2 grid of 3x3 blocks
    block_size = 3
    num_blocks = 2
    
    # Strassen coefficients for 2x2 block multiplication
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # For each Strassen multiplication q in 0..6:
    # alpha[q][a_block][b_block] = coefficient in A combination
    # beta[q][c_block][d_block] = coefficient in B combination
    # gamma[q][out_block] = coefficient for output C block
    
    # A block indices: (row, col) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B block indices: (row, col) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C block indices: (row, col) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    alpha = [
        # q=0: (A00+A11)*(B00+B11)
        {0: 1, 3: 1},
        # q=1: (A10+A11)*B00
        {2: 1, 3: 1},
        # q=2: A00*(B01-B11)
        {0: 1},
        # q=3: A11*(B10-B00)
        {3: 1},
        # q=4: (A00+A01)*B11
        {0: 1, 1: 1},
        # q=5: (A10-A00)*(B00+B01)
        {2: 1, 0: -1},
        # q=6: (A01-A11)*(B10+B11)
        {1: 1, 3: -1},
    ]
    
    beta = [
        # q=0: (A00+A11)*(B00+B11)
        {0: 1, 3: 1},
        # q=1: (A10+A11)*B00
        {0: 1},
        # q=2: A00*(B01-B11)
        {1: 1, 3: -1},
        # q=3: A11*(B10-B00)
        {2: 1, 0: -1},
        # q=4: (A00+A01)*B11
        {3: 1},
        # q=5: (A10-A00)*(B00+B01)
        {0: 1, 1: 1},
        # q=6: (A01-A11)*(B10+B11)
        {2: 1, 3: 1},
    ]
    
    # C00 = M0 + M3 - M5 + M6
    # C01 = M2 + M5
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M4
    gamma = [
        # q=0: contributes to C00, C11
        {0: 1, 3: 1},
        # q=1: contributes to C10, C11
        {2: 1, 3: -1},
        # q=2: contributes to C01, C11
        {1: 1, 3: 1},
        # q=3: contributes to C00, C10
        {0: 1, 2: 1},
        # q=4: contributes to C11
        {3: 1},
        # q=5: contributes to C00, C01
        {0: -1, 1: 1},
        # q=6: contributes to C00
        {0: 1},
    ]
    
    # Each Strassen multiplication involves block_size^3 = 27 scalar multiplications
    rank = 7 * block_size * block_size * block_size
    rank = 7 * 27
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    col = 0
    for q in range(7):
        for u in range(block_size):  # row index within A block
            for w in range(block_size):  # col index within B block (which becomes row in W)
                for v in range(block_size):  # col index within A block / row in B block
                
                    # U indices: for each A block (ai, aj), element (u, v) maps to row ai*6 + u*3 + v
                    for ai in range(2):
                        for aj in range(2):
                            a_block = ai * 2 + aj
                            coeff = alpha[q].get(a_block, 0)
                            if coeff != 0:
                                row_u = ai * (block_size * 2) + u * block_size + v
                                U[row_u][col] += coeff
                    
                    # V indices: for each B block (bi, bj), element (u, w) maps to row bi*6 + u*3 + w
                    # Wait, B is (m, p) = (6, 6). B block (bi, bj) covers rows bi*3..bi*3+2, cols bj*3..bj*3+2
                    # V row index: bi*6 + (row within block)*3 + (col within block)
                    # The B block element that pairs with A block element (u, v) is (v, w)
                    for bi in range(2):
                        for bj in range(2):
                            b_block = bi * 2 + bj
                            coeff = beta[q].get(b_block, 0)
                            if coeff != 0:
                                row_v = bi * (block_size * 2) + v * block_size + w
                                V[row_v][col] += coeff
                    
                    # W indices: for each C block (ci, cj), element (u, w) maps to row ci*6 + u*3 + w
                    # W[k*n+i, r] convention: k is row of result, i is col of result
                    # C block (ci, cj) covers rows ci*3..ci*3+2, cols cj*3..cj*3+2
                    # W row index: ci*6 + u*3 + w... wait let me re-examine
                    # W[k*n+i] where k = output row, i = output col
                    # For C block (ci, cj), element (u, w): k = ci*3+u, i = cj*3+w
                    # So W row = k*6 + i = (ci*3+u)*6 + cj*3 + w
                    for ci in range(2):
                        for cj in range(2):
                            c_block = ci * 2 + cj
                            coeff = gamma[q].get(c_block, 0)
                            if coeff != 0:
                                row_w = (ci * block_size + u) * n + cj * block_size + w
                                W[row_w][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
