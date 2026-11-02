"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    We use a recursive block-Strassen approach:
    - Split the 4x4 matrices into 2x2 blocks
    - Use Strassen's algorithm (rank 7) for 2x2 * 2x2 multiplications
    - Handle the rectangular 4x5 case by splitting into 2x2 and 2x3 blocks
    
    For 4x4 * 4x5:
    A = [[A11, A12], [A21, A22]] where each block is 2x2
    B = [[B11, B12], [B21, B22]] where B11, B21 are 2x2 and B12, B22 are 2x3
    
    C = A*B where C is 4x5:
    C11 = A11*B11 + A12*B21 (2x2)
    C12 = A11*B12 + A12*B22 (2x3)
    C21 = A21*B11 + A22*B21 (2x2)
    C22 = A21*B12 + A22*B22 (2x3)
    
    For the 2x2 * 2x2 products, we can use Strassen (rank 7 each).
    For the 2x2 * 2x3 products, we need a decomposition.
    
    Actually, let's use a simpler approach: use the known Strassen decomposition
    for 2x2 matrices and extend it to handle the rectangular case.
    
    Strassen's 7-multiplication algorithm for 2x2:
    m1 = (A11+A22)*(B11+B22)
    m2 = (A21+A22)*B11
    m3 = A11*(B12-B22)
    m4 = A22*(B21-B11)
    m5 = (A11+A12)*B22
    m6 = (A21-A11)*(B11+B12)
    m7 = (A12-A22)*(B21+B22)
    
    C11 = m1 + m4 - m5 + m7
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 - m2 + m3 + m6
    
    For 2x2 * 2x3, we can pad the 2x3 to 2x4 and use a variant, or use a direct
    low-rank decomposition.
    
    Let's construct the decomposition using the tensor rank approach with
    integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # We'll use a block decomposition approach
    # Split into 2x2 blocks for A and B
    # A is 4x4, B is 4x5
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01], [B10, B11]] where B00, B10 are 2x2, B01, B11 are 2x3
    
    # C = A*B is 4x5
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10 (2x2)
    # C01 = A00*B01 + A01*B11 (2x3)
    # C10 = A10*B00 + A11*B10 (2x2)
    # C11 = A10*B01 + A11*B11 (2x3)
    
    # For 2x2*2x2, use Strassen (rank 7)
    # For 2x2*2x3, we need a decomposition
    
    # Let's construct the full decomposition
    # Total rank = 4 * 7 (for the four 2x2*2x2 blocks) + rank for 2x2*2x3 terms
    
    # Actually, let's use a different approach: construct the decomposition
    # directly using the tensor structure.
    
    # For now, let's return the schoolbook decomposition as a baseline
    # and try to improve it.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
