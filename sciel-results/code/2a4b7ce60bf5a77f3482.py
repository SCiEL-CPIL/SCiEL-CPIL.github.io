"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from scipy.optimize import linprog

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 7
    rank_schoolbook = n * m * p  # 63
    
    # Schoolbook decomposition
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for i2 in range(n):
                    for j2 in range(m):
                        for k in range(p):
                            for k2 in range(p):
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                actual = 0.0
                                for r in range(rank):
                                    actual += U[i*m+j][r] * V[j2*p+k2][r] * W[k*n+i][r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    # Strategy: Try to find a lower-rank decomposition using LP over finite fields
    # to identify sparse support patterns, then verify exact real-valued solutions.
    
    # For now, let's try a few lower ranks with a structured approach.
    # We'll use the fact that the tensor has a diagonal structure and try to
    # find decompositions with half-integer or integer entries.
    
    # Try ranks from 55 down to 40, looking for valid decompositions
    for target_rank in range(55, 39, -1):
        if time.time() > deadline - 20.0:
            break
        
        # For each target rank, try to find a decomposition
        # We'll use a greedy approach: start with schoolbook and try to merge columns
        
        # Create a copy of schoolbook
        U_try = [row[:] for row in U_school[:target_rank]]
        V_try = [row[:] for row in V_school[:target_rank]]
        W_try = [row[:] for row in W_school[:target_rank]]
        
        # Try to verify if this truncated version works (it won't, but let's check)
        # Instead, let's try a different approach: use random sparse patterns
        
        for attempt in range(5):
            if time.time() > deadline - 20.0:
                break
            
            # Generate random sparse matrices with half-integer entries
            U_rand = []
            for _ in range(n * m):
                row = []
                for _ in range(target_rank):
                    if random.random() < 0.3:  # 30% sparsity
                        row.append(random.choice([-1, -0.5, 0.5, 1]))
                    else:
                        row.append(0)
                U_rand.append(row)
            
            V_rand = []
            for _ in range(m * p):
                row = []
                for _ in range(target_rank):
                    if random.random() < 0.3:
                        row.append(random.choice([-1, -0.5, 0.5, 1]))
                    else:
                        row.append(0)
                V_rand.append(row)
            
            W_rand = []
            for _ in range(p * n):
                row = []
                for _ in range(target_rank):
                    if random.random() < 0.3:
                        row.append(random.choice([-1, -0.5, 0.5, 1]))
                    else:
                        row.append(0)
                W_rand.append(row)
            
            # Verify
            if verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                best_rank = target_rank
                break
    
    if time.time() > deadline - 5.0:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
