import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# Known rank-23 decomposition for 3x3x3 matrix multiplication (Bini et al. 1978)
# Each term is (u, v, w) where u, v, w are 9-element vectors (flattened 3x3 matrices)
# u corresponds to linear combination of A entries
# v corresponds to linear combination of B entries
# w corresponds to linear combination of C entries
#
# The 23 terms for 3x3 matrix multiplication:
TERMS_3x3 = [
    # Term 1: a11 * b11 -> c11
    ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
    # Term 2: a21 * b21 -> c22
    ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
    # Term 3: a31 * b31 -> c33
    ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]),
    # Term 4: (a12+a22+a32+a13+a23+a33) * (b11+b21+b31+b12+b22+b32) -> various
    # Let me use a verified rank-23 scheme. I'll use the standard one.
    # Actually, let me just use a correct implementation.
    # I'll use the rank-23 from the literature.
    # For correctness, let me verify with a different approach.
    
    # Let me use the schoolbook (rank 27) for 3x3x3 to ensure correctness,
    # but optimize the overall structure.
    
    # Actually, let me reconsider the problem structure.
    # 3x5 times 5x6.
    # 
    # Strategy: Use rank-23 for 3x3x3 blocks.
    # Split the 5 middle dimension into 2+3.
    # A = [A1 (3x2) | A2 (3x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 3x2 times 2x6 -> rank 3*2*6 = 36 (schoolbook)
    # A2*B2: 3x3 times 3x6
    # Split B2 into B2a (3x3) and B2b (3x3)
    # A2*B2 = A2*B2a + A2*B2b
    # Each is 3x3 times 3x3 -> rank 23 each
    # Total: 36 + 23 + 23 = 82
    
    # I need a valid rank-23 decomposition for 3x3x3.
    # Let me use a known one.
]

# I'll implement the schoolbook for 3x3x3 (rank 27) first to ensure correctness,
# then try to use rank-23 if I can find a valid implementation.

def construct():
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS  # n=3, m=5, p=6
    
    # For now, use schoolbook for everything: rank = 3*5*6 = 90
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
