"""
Program implementing Strassen-based rank minimization for 4x4x6 matrix multiplication.

Uses the optimal 49-term Strassen decomposition for the 4x4x4 part and
a 28-term decomposition for the 4x4x2 part, for a total rank of 77.
"""
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 terms
    strassen_2x2_terms = [
        {'A': [[1, 0], [0, 1]], 'B': [[1, 0], [0, 1]], 'C': [[1, 0], [0, 1]]},
        {'A': [[0, 0], [1, 1]], 'B': [[1, 0], [0, 0]], 'C': [[0, 0], [1, -1]]},
        {'A': [[1, 0], [0, 0]], 'B': [[0, 1], [0, -1]], 'C': [[0, 1], [0, 1]]},
        {'A': [[0, 0], [0, 1]], 'B': [[-1, 0], [1, 0]], 'C': [[1, 0], [1, 0]]},
        {'A': [[1, 1], [0, 0]], 'B': [[0, 0], [0, 1]], 'C': [[-1, 1], [0, 0]]},
        {'A': [[-1, 0], [1, 0]], 'B': [[1, 1], [0, 0]], 'C': [[0, 0], [0, 1]]},
        {'A': [[0, 1], [0, -1]], 'B': [[0, 0], [1, 1]], 'C': [[1, 0], [0, 0]]},
    ]
    
    terms = []
    
    def add_product_of_block_sums(a_blocks, b_blocks, c_blocks):
        """
        Add 7 Strassen terms for the product of linear combinations of 2x2 blocks.
        a_blocks: list of (row_offset, col_offset, coeff)
        b_blocks: list of (row_offset, col_offset, coeff)
        c_blocks: list of (row_offset, col_offset, coeff)
        """
        for t in strassen_2x2_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in a_blocks:
                        u[(r + i) * m + (c + j)] += coeff * t['A'][i][j]
            
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in b_blocks:
                        v[(r + i) * p + (c + j)] += coeff * t['B'][i][j]
            
            for i in range(2):
                for j in range(2):
                    for (r, c, cc) in c_blocks:
                        w[(c + j) * n + (r + i)] += cc * t['C'][i][j]
            
            terms.append((u, v, w))
    
    # 4x4x4 part: 7 Strassen products, each decomposed into 7 Strassen 2x2 terms = 49
    # M1 = (A00+A11)(B00+B11) -> C00 (+1), C11 (+1)
    add_product_of_block_sums(
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)]
    )
    
    # M2 = (A10+A11)B00 -> C10 (+1), C11 (-1)
    add_product_of_block_sums(
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)],
        [(2, 0, 1), (2, 2, -1)]
    )
    
    # M3 = A00(B01-B11) -> C01 (+1), C11 (+1)
    add_product_of_block_sums(
        [(0, 0, 1)],
        [(0, 2, 1), (2, 2, -1)],
        [(0, 2, 1), (2, 2, 1)]
    )
    
    # M4 = A11(B10-B00) -> C00 (+1), C10 (+1)
    add_product_of_block_sums(
        [(2, 2, 1)],
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (2, 0, 1)]
    )
    
    # M5 = (A00+A01)B11 -> C00 (-1), C01 (+1)
    add_product_of_block_sums(
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)],
        [(0, 0, -1), (0, 2, 1)]
    )
    
    # M6 = (A10-A00)(B00+B01) -> C11 (+1)
    add_product_of_block_sums(
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)]
    )
    
    # M7 = (A01-A11)(B10+B11) -> C00 (+1)
    add_product_of_block_sums(
        [(0, 2, 1), (2, 2, -1)],
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)]
    )
    
    # 4x4x2 part: 4 block products, each 7 terms = 28
    # C_top = A00*B_top + A01*B_bottom
    add_product_of_block_sums(
        [(0, 0, 1)],
        [(0, 4, 1)],
        [(0, 4, 1)]
    )
    add_product_of_block_sums(
        [(0, 2, 1)],
        [(2, 4, 1)],
        [(0, 4, 1)]
    )
    # C_bottom = A10*B_top + A11*B_bottom
    add_product_of_block_sums(
        [(2, 0, 1)],
        [(0, 4, 1)],
        [(2, 4, 1)]
    )
    add_product_of_block_sums(
        [(2, 2, 1)],
        [(2, 4, 1)],
        [(2, 4, 1)]
    )
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][k] = u[i]
        for i in range(m * p):
            V[i][k] = v[i]
        for i in range(p * n):
            W[i][k] = w[i]
    
    return {"U": U, "V": V, "W": W}
