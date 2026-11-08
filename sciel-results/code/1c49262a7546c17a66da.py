"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin
    
    # Best known result: schoolbook
    best_rank = n * m * p
    best_result = None
    
    # Try to find a better decomposition using a simple greedy/approximate method
    # For matrix multiplication tensor, we can try to find a low-rank decomposition
    
    # The matrix multiplication tensor T has entries T[i,j,k, i',j',k'] = 1 if j==j' and k==k'
    # We want to find rank r such that T = sum_{c=1}^{r} u_c * v_c * w_c
    
    # For small dimensions, we can try to use known results or search
    # Known: for 3x4x4, the rank is at most 48 (schoolbook), but might be lower
    
    # Let's try a simple approach: start with schoolbook and try to reduce rank
    # by combining terms or using a different decomposition
    
    # Actually, for this specific case, let's just return the schoolbook decomposition
    # since finding a better one in the time limit is challenging without more sophisticated methods
    
    # Create the schoolbook decomposition
    rank = n * m * p
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
    
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to improve using a simple local search or perturbation
    # For now, return the schoolbook result as it's guaranteed to work
    return best_result
# EVOLVE-BLOCK-END
