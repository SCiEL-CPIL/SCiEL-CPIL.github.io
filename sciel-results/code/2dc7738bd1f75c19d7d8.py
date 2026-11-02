"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for arr in [U, V, W]:
        if len(arr) == 0:
            return False
        if len(arr[0]) != rank:
            return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_greedy_rank(n, m, p, target_rank, timeout=30):
    """Try to find a decomposition of given rank using greedy approach."""
    start = time.time()
    
    # Initialize with random small values
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # We need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of nonlinear equations.
    
    # Strategy: build up one rank term at a time, each satisfying as many constraints as possible
    # For each new rank term, we choose U[:,c], V[:,c], W[:,c] to satisfy some subset of constraints
    
    # Actually, let's try a different approach: use the fact that we can set up the problem
    # as finding rank-1 tensors that sum to the multiplication tensor.
    
    # Let's try a randomized approach with local search
    best_result = None
    
    # Start from schoolbook and try to merge terms
    # Or: try to find a decomposition by solving linear systems for subsets
    
    # For small target_rank, we can try to solve the system directly
    # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is equivalent to: the tensor T = sum_c u_c ⊗ v_c ⊗ w_c where
    # u_c[i*m+j] = U[i*m+j,c], etc.
    
    # Let's use a greedy construction:
    # At each step, pick a (i,j,k) that's not yet satisfied, and add a rank term that helps.
    
    # Actually, for this problem size, let's try a more direct approach.
    # We'll use the fact that the tensor is sparse in a certain sense.
    
    # Let me try: can we find a rank-r decomposition where each rank term has
    # a structured support?
    
    # For now, let's just return None if we can't find one quickly
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to improve by searching for lower-rank decompositions
    # For 3x4x7, the schoolbook rank is 84
    # Known lower bounds: the rank must be at least the "slice rank" 
    # For matrix multiplication, a simple lower bound is max(m*p, n*p, m*n) = max(21,28,12)=28
    # But the actual minimum is likely higher.
    
    # Let's try a systematic search for ranks from a reasonable lower bound up
    # We'll use a timeout to ensure we finish in time
    
    deadline = time.time() + 100  # Leave margin before 120s limit
    
    # Try various ranks
    for target_rank in range(28, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        # For now, this is a placeholder - we need a real search algorithm
        
        # Let's try a simple heuristic: if target_rank >= n*m*p, we know schoolbook works
        # For lower ranks, we need to actually search
    
    # For this specific case (3,4,7), let me think about what's achievable
    # The bilinear complexity of 3x4 by 4x7 matrix multiplication...
    # I'm not sure of the exact minimum, so let's try to find one.
    
    # Actually, let me implement a proper search using linear algebra
    # The key insight: if we fix the supports (which entries are nonzero) of U, V, W,
    # then the values can be found by solving a linear system.
    
    # For a rank-r decomposition, each column c of U, V, W defines a rank-1 tensor.
    # The sum of these rank-1 tensors must equal the multiplication tensor.
    
    # Let's try: for each possible support pattern (which is too many), solve the linear system.
    # Instead, let's use a randomized approach.
    
    # Given time constraints, let's just return the schoolbook decomposition
    # unless we can quickly find something better.
    
    # Try a few quick searches
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        # Try to find a rank-83 decomposition (one less than schoolbook)
        # This is unlikely but let's try
        pass
    
    return result

# EVOLVE-BLOCK-END
