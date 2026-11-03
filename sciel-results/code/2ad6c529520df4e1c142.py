import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    # Try to find a better decomposition using MILP-like approach
    # For now, return the trivial solution as it's guaranteed to work
    return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
