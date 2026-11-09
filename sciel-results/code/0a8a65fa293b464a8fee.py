import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    # Schoolbook
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using a simple heuristic
    # This is a placeholder for a more complex search
    # For now, return schoolbook to ensure validity
    return {"U": U, "V": V, "W": W}
