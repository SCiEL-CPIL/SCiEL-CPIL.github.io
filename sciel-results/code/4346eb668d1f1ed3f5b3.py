<<<<<<< SEARCH
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is sum of ranks of parts
    # Part 1: 2x2x5 -> 7+7+4 = 18
    # Part 2: 2x3x5 -> 11+11+6 = 28
    # Total 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1
=======
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is sum of ranks of parts
    # Part 1: 2x2x5 -> 7+7+4 = 18
    # Part 2: 2x3x5 -> 11+11+6 = 28
    # Total 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = [0]
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        U[i * m + j][col[0]] = u_val
        V[j * p + k][col[0]] = v_val
        W[k * n + i][col[0]] = w_val
        col[0] += 1
>>>>>>> REPLACE

