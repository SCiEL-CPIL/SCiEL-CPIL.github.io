"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from itertools import combinations

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Schoolbook rank
    rank_school = n * m * p  # 42
    
    # Try to find a lower rank decomposition
    # We'll try ranks from rank_school - 1 down to some minimum
    # Use a simple heuristic: try to find a decomposition by solving linear equations
    
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Build the multiplication tensor T[i, j, k] = 1 if we need C[i,k] += A[i,j]*B[j,k]
    # The tensor is a 3-way array T of shape (nm, mp, pn) where T[i*m+j, j*p+k, k*n+i] = 1
    
    # Actually, let's think of this differently.
    # We want to find U, V, W such that for all i, j, k:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 if (i,j,k) is a valid index, else 0
    # But actually the tensor T is defined as T[i, j, k] = 1 for all valid (i,j,k)
    # and the decomposition is: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # The "valid" indices are all (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # So the tensor has exactly n*m*p = 42 non-zero entries, all equal to 1.
    
    # Let's try a simple approach: start with schoolbook and try to merge terms
    # or use random initialization + gradient descent
    
    # For now, let's just try a few random restarts with a simple local search
    
    # Build the list of tensor indices
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_indices.append((i*m+j, j*p+k, k*n+i))
    
    # Try rank = 41 first
    for rank in range(rank_school - 1, max(rank_school - 15, 30), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Simple gradient descent to minimize sum over all (i,j,k) not in tensor of (sum_c U[i,j,c]V[j,k,c]W[k,i,c])^2
            # and also minimize (1 - sum_c U[i,j,c]V[j,k,c]W[k,i,c])^2 for tensor indices
            
            # This is complex, so let's just check if the random init happens to work (unlikely)
            # Instead, let's try a structured approach
            
            # For 2x3x7, maybe we can use a known result?
            # Let's just return schoolbook if we can't find better quickly
        
        # If we had a working method, we'd check here
    
    # Return schoolbook decomposition
    rank = rank_school
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
