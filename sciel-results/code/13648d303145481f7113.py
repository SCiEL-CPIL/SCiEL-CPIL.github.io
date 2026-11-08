"""Program attempting to find a rank-minimal decomposition for 2x3x9 matrix multiplication."""
import time
import sys
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # Keep a safety margin from the 150s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Baseline: Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k
                U_sb[i * m + j][idx] = 1
                V_sb[j * p + k][idx] = 1
                W_sb[k * n + i][idx] = 1
                
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Build the target tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need to find U, V, W such that:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Strategy: Try to find lower-rank decompositions using structured search
    # Start from rank 27 (theoretical lower bound) and go up
    
    # For small ranks, we can use a randomized approach with integer/half-integer entries
    # from a small set of values
    
    values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    # Try ranks from lower bound upward
    for target_rank in range(27, min(rank_sb + 1, 40)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations for this rank
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization with small integer/half-integer values
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with random values from our set
            for idx in range(n * m):
                for r in range(target_rank):
                    U[idx, r] = random.choice(values)
            for idx in range(m * p):
                for r in range(target_rank):
                    V[idx, r] = random.choice(values)
            for idx in range(p * n):
                for r in range(target_rank):
                    W[idx, r] = random.choice(values)
            
            # Check if this random assignment works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists
                U_list = [[float(U[i, r]) for r in range(target_rank)] for i in range(n * m)]
                V_list = [[float(V[i, r]) for r in range(target_rank)] for i in range(m * p)]
                W_list = [[float(W[i, r]) for r in range(target_rank)] for i in range(p * n)]
                
                # Verify once more
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                    if not valid:
                        break
                
                if valid:
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    return best
        
        # If we found a valid decomposition at this rank, return it
        # (the inner loop already returns when found)
    
    # If no better decomposition found, return the schoolbook baseline
    return best
# EVOLVE-BLOCK-END
