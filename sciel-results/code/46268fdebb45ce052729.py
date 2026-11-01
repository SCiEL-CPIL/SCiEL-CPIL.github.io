"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses Strassen's algorithm on 6 independent 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    num_pairs = p // 2  # 6
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                U[0][c] = 1; U[3][c] = 1
                V[0*p+2*t][c] = 1; V[1*p+2*t+1][c] = 1
                W[2*t*n+0][c] = 1; W[(2*t+1)*n+1][c] = 1
            elif s == 1:
                U[2][c] = 1; U[3][c] = 1
                V[0*p+2*t][c] = 1
                W[2*t*n+1][c] = 1; W[(2*t+1)*n+1][c] = -1
            elif s == 2:
                U[0][c] = 1
                V[0*p+2*t+1][c] = 1; V[1*p+2*t+1][c] = -1
                W[(2*t+1)*n+0][c] = 1; W[(2*t+1)*n+1][c] = 1
            elif s == 3:
                U[3][c] = 1
                V[1*p+2*t][c] = 1; V[0*p+2*t][c] = -1
                W[2*t*n+0][c] = 1; W[2*t*n+1][c] = 1
            elif s == 4:
                U[0][c] = 1; U[1][c] = 1
                V[1*p+2*t+1][c] = 1
                W[2*t*n+0][c] = -1; W[(2*t+1)*n+0][c] = 1
            elif s == 5:
                U[2][c] = 1; U[0][c] = -1
                V[0*p+2*t][c] = 1; V[0*p+2*t+1][c] = 1
                W[(2*t+1)*n+1][c] = 1
            elif s == 6:
                U[1][c] = 1; U[3][c] = -1
                V[1*p+2*t][c] = 1; V[1*p+2*t+1][c] = 1
                W[2*t*n+0][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
