import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 7]

def verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is valid."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for t in range(rank):
                    total += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if total != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # Build trivial decomposition (rank 42)
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_solution = {"U": U, "V": V, "W": W}
    best_rank = trivial_rank

    # Try to find lower rank decompositions using structured approaches
    # For 2x3x7, try ranks from 41 down to 35
    
    # Approach: Try random integer/half-integer matrices and verify
    # This is a heuristic search
    
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if target_rank > 38 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random matrices with small integer entries
            U_r = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V_r = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W_r = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify
            if verify_decomposition(U_r, V_r, W_r, n, m, p, target_rank):
                best_rank = target_rank
                best_solution = {"U": U_r, "V": V_r, "W": W_r}
                break
        
        if best_rank < trivial_rank:
            break
    
    return {"U": best_solution["U"], "V": best_solution["V"], "W": best_solution["W"]}
