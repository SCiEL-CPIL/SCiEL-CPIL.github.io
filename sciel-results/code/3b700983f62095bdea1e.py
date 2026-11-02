"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[Fraction(0) for _ in range(rank)] for _ in range(n * m)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(m * p)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
