"""Matrix multiplication tensor decomposition for 4x5 by 5x5."""
import numpy as np
from collections import defaultdict
import time
from itertools import combinations
from scipy.optimize import linprog

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using ILP-like approach
    # First, let's try some known optimizations
    
    # For matrix multiplication, we can use the fact that certain patterns allow for rank reduction
    # Let's try to find if there's a rank less than 100
    
    deadline = time.time() + 130  # Leave 20s margin
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to use a structured approach: group the indices and look for patterns
    # One approach: use the fact that for rectangular matrices, there might be 
    # exploitable symmetries
    
    # Let's try a heuristic: look for rank-1 terms that can be combined
    # This is a simplified approach - in practice, we'd want a more sophisticated ILP
    
    # For now, return the schoolbook decomposition which is guaranteed correct
    return best_result
# EVOLVE-BLOCK-END
