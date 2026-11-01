"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook rank
    best_rank = n * m * p  # 42
    best_decomp = _schoolbook_decomp(n, m, p, best_rank)
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            decomp = _try_rank_search(n, m, p, target_rank, attempt)
            if decomp is not None:
                best_rank = target_rank
                best_decomp = decomp
                break
    
    return best_decomp

def _schoolbook_decomp(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _try_rank_search(n, m, p, target_rank, seed):
    """Try to find a rank-target_rank decomposition using structured search."""
    random.seed(seed * 1000 + target_rank)
    
    U = [[0] * target_rank for _ in range(n*m)]
    V = [[0] * target_rank for _ in range(m*p)]
    W = [[0] * target_rank for _ in range(p*n)]
    
    # Initialize with a structured pattern
    direct_terms = min(target_rank, n * m * p)
    for r in range(direct_terms):
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    # Local search to find a valid decomposition
    for iteration in range(300):
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Modify a random entry
        matrix_choice = random.randint(0, 2)
        col = random.randint(0, target_rank-1)
        if matrix_choice == 0:
            idx = random.randint(0, n*m-1)
            U[idx][col] = random.randint(-3, 3)
        elif matrix_choice == 1:
            idx = random.randint(0, m*p-1)
            V[idx][col] = random.randint(-3, 3)
        else:
            idx = random.randint(0, p*n-1)
            W[idx][col] = random.randint(-3, 3)
    
    return None
# EVOLVE-BLOCK-END
