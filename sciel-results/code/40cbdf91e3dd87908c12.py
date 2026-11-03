"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 4x4 + 1x1
    # Rank of 4x4 Strassen is 49.
    # Rank of 4x1, 1x4, 1x1 terms are 4, 4, 1 respectively.
    # Total rank = 49 (C44) + 49 (C44 cross) + 4 (C41) + 4 (C41 cross) + 4 (C14) + 4 (C14 cross) + 1 (C11) + 1 (C11 cross)
    # Wait, the formula is:
    # C44 = A44 B44 + A41 B14
    # C41 = A44 B41 + A41 B11
    # C14 = A14 B44 + A11 B14
    # C11 = A14 B41 + A11 B11
    
    # We need to compute:
    # A44 B44 (Rank 49)
    # A41 B14 (A41 is 4x1, B14 is 1x4, Rank 4)
    # A44 B41 (A44 is 4x4, B41 is 4x1, Rank 4) -- Wait, B41 is 4x1? No, B is 5x5.
    # Let's index:
    # A = [[A11, A12, A13, A14, A15], ...]
    # Block 1: 4x4, Block 2: 4x1, Block 3: 1x4, Block 4: 1x1
    # A11 (4x4), A12 (4x1), A13 (1x4), A14 (1x1)
    # B11 (4x4), B12 (4x1), B13 (1x4), B14 (1x1)
    
    # C11 = A11 B11 + A12 B13
    # C12 = A11 B12 + A12 B14
    # C13 = A13 B11 + A14 B13
    # C14 = A13 B12 + A14 B14
    
    # Ranks:
    # A11 B11: 49
    # A12 B13: 4x1 * 1x4 -> 4
    # A11 B12: 4x4 * 4x1 -> 4 (outer product of 4 vectors? No, rank of 4x4x1 tensor is 4)
    # A12 B14: 4x1 * 1x1 -> 4
    # A13 B11: 1x4 * 4x4 -> 4
    # A14 B13: 1x1 * 1x4 -> 4
    # A13 B12: 1x4 * 4x1 -> 4
    # A14 B14: 1x1 * 1x1 -> 1
    
    # Total rank = 49 + 4 + 4 + 4 + 4 + 4 + 4 + 1 = 74?
    # Wait, A11 B12 is 4x4 * 4x1 = 4x1. The rank of this tensor is 4.
    # So total rank is 49 + 4*7 = 77?
    # Let's re-calculate.
    # A11 B11: 49
    # A12 B13: 4
    # A11 B12: 4
    # A12 B14: 4
    # A13 B11: 4
    # A14 B13: 4
    # A13 B12: 4
    # A14 B14: 1
    # Total = 49 + 24 + 1 = 74.
    
    # This is much better than 125!
    
    # I need to implement the Strassen 4x4 decomposition.
    # And the simple decompositions for the other terms.
    
    # For now, I will return the schoolbook to ensure correctness, as implementing Strassen 4x4 correctly is complex.
    # But 74 is a big jump.
    # I will try to implement it.
    
    rank = 125
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
