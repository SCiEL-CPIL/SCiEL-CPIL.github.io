"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s out of 150s limit
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # For 3x5x6 matrix multiplication, we can try to find a decomposition
    # with rank less than 90.
    
    # One approach: use the fact that matrix multiplication is bilinear and
    # try to find a structured decomposition.
    
    # For small matrices, we can try to exploit the structure.
    # For example, for 2x2x2, Strassen's algorithm gives rank 7.
    # For 3x3x3, the best known is 23.
    # For 3x5x6, let's try to find a good decomposition.
    
    # Strategy: Use random search with structured initialization.
    # We'll try ranks from a reasonable lower bound up to schoolbook.
    
    # Lower bound: The tensor has rank at least max(n, m, p) * something.
    # A simple lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30.
    # But we can do better.
    
    # Let's try a few ranks with random search.
    random.seed(42)
    
    # Try ranks from 40 to 90 (skip schoolbook since we already have it)
    # We'll try a few random initializations for each rank.
    
    ranks_to_try = list(range(40, 90))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        if rank >= best_rank:
            continue
        
        # Try multiple random initializations
        max_attempts = 30 if rank < 60 else 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U, V, W with integer or half-integer entries
            # Use small values to increase chance of exactness
            U_rand = np.zeros((n * m, rank), dtype=float)
            V_rand = np.zeros((m * p, rank), dtype=float)
            W_rand = np.zeros((p * n, rank), dtype=float)
            
            # Fill with random small integers or half-integers
            for i in range(n * m):
                for c in range(rank):
                    if random.random() < 0.4:  # 40% chance of nonzero
                        val = random.choice([-1, -0.5, 0.5, 1])
                        U_rand[i, c] = val
            
            for j in range(m * p):
                for c in range(rank):
                    if random.random() < 0.4:
                        val = random.choice([-1, -0.5, 0.5, 1])
                        V_rand[j, c] = val
            
            for k in range(p * n):
                for c in range(rank):
                    if random.random() < 0.4:
                        val = random.choice([-1, -0.5, 0.5, 1])
                        W_rand[k, c] = val
            
            # Check if this decomposes correctly
            if check_decomposition(U_rand, V_rand, W_rand, rank):
                # Convert to lists with exact integer/half-integer representation
                U_list = [[int(U_rand[i, c]) if abs(U_rand[i, c] - round(U_rand[i, c])) < 1e-9 
                          else U_rand[i, c] for c in range(rank)] for i in range(n * m)]
                V_list = [[int(V_rand[j, c]) if abs(V_rand[j, c] - round(V_rand[j, c])) < 1e-9 
                          else V_rand[j, c] for c in range(rank)] for j in range(m * p)]
                W_list = [[int(W_rand[k, c]) if abs(W_rand[k, c] - round(W_rand[k, c])) < 1e-9 
                          else W_rand[k, c] for c in range(rank)] for k in range(p * n)]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                print(f"Found rank {rank} decomposition!")
                break  # Move to next (lower) rank
        
        if rank < best_rank:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
