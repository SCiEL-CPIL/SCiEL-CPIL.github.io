"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_school}
    
    # Try to find better decomposition using random search + local optimization
    # This is a heuristic search for lower rank
    
    # We'll use a simple approach: try random initializations and see if we can find
    # a valid decomposition at lower rank.
    
    # For now, just return schoolbook as a safe baseline.
    # A more sophisticated search would be added here if time permits.
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
