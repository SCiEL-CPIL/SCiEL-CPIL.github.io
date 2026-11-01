import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients - will verify and fix if needed
    a_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
    ]
    b_coeffs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, -1], [0, 1]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ]
    c_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
    ]
    
    for jg in range(2):
        for kg in range(4):
            for s in range(7):
                col = jg * 28 + kg * 7 + s
                for i in range(2):
                    for jl in range(2):
                        j = jg * 2 + jl
                        U[i * m + j][col] = a_coeffs[s][i][jl]
                for jl in range(2):
                    for kl in range(2):
                        j = jg * 2 + jl
                        k = kg * 2 + kl
                        V[j * p + k][col] = b_coeffs[s][jl][kl]
                for i in range(2):
                    for kl in range(2):
                        k = kg * 2 + kl
                        W[k * n + i][col] = c_coeffs[s][i][kl]
    
    return {"U": U, "V": V, "W": W}
