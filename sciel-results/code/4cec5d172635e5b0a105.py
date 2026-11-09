"""
Matrix multiplication 3x3 * 3x4 tensor decomposition.
Returns the naive rank-36 decomposition.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Naive rank is n * m * p
    rank = n * m * p
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U index: i * m + k
                # V index: k * p + j
                # W index: j * n + i
                U[i * m + k][col] = Fraction(1)
                V[k * p + j][col] = Fraction(1)
                W[j * n + i][col] = Fraction(1)
                col += 1
    
    # Convert to lists of floats/ints for output
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    # Clean up floats to ints if possible
    def convert_to_clean(mat):
        mat_out = []
        for row in mat:
            row_out = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_out.append(int(round(val)))
                else:
                    row_out.append(val)
            mat_out.append(row_out)
        return mat_out
    
    U_out = convert_to_clean(U_out)
    V_out = convert_to_clean(V_out)
    W_out = convert_to_clean(W_out)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
