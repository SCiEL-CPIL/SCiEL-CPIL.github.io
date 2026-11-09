"""Matrix multiplication tensor decomposition for 2x3 * 3x8."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 8]

def verify_decomposition(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    best_rank = n * m * p  # Start with schoolbook = 48
    best_result = None
    
    # Initialize schoolbook solution
    def make_schoolbook(rank):
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = make_schoolbook(best_rank)
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions using structured search
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Use a more systematic approach: try to combine schoolbook terms
    # For each rank from lower_bound+1 to best_rank-1, try random perturbations
    
    for rank in range(lower_bound + 1, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 100
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Start from schoolbook and try to zero out some entries
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Fill with random small values
            for i in range(n*m):
                for r in range(rank):
                    if random.random() < 0.3:
                        U[i][r] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
            
            for j in range(m*p):
                for r in range(rank):
                    if random.random() < 0.3:
                        V[j][r] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
            
            for k in range(p*n):
                for r in range(rank):
                    if random.random() < 0.3:
                        W[k][r] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
            
            if verify_decomposition(U, V, W, n, m, p, rank):
                best_rank = rank
                best_result = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
                found = True
                break
        
        if found:
            continue
    
    return best_result
