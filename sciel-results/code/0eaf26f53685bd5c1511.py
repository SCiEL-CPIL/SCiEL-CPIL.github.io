import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def make_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    return make_rank28()
# EVOLVE-BLOCK-END
