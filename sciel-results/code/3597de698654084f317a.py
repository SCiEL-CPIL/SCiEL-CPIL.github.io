import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40

    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W):
        T_approx = np.einsum('ut,vt,wt->uvw', U, V, W)
        return np.allclose(T_approx, T_target, atol=1e-9)

    # Schoolbook solution (Rank 40)
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    best_solution = {'U': U_school, 'V': V_school, 'W': W_school}
    best_rank = rank_schoolbook

    # Strategy: Try to find lower rank by exploiting structure
    # For 2x4x5, try ranks from 39 down
    
    # Approach: Random search with integer/half-integer entries
    # Use a smarter initialization based on the structure
    
    def try_random_rank(r, max_attempts=200):
        """Try to find a rank-r decomposition with random integer/half-integer entries"""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            # Use values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U = np.random.choice(vals, size=(n*m, r))
            V = np.random.choice(vals, size=(m*p, r))
            W = np.random.choice(vals, size=(p*n, r))
            
            if verify(U, V, W):
                return U, V, W
        
        return None

    # Approach: Structured search using the block decomposition idea
    # The 2x4x5 tensor can potentially be decomposed using sub-structures
    
    def structured_search(r):
        """Try structured decompositions based on block structure"""
        if time.time() > deadline - 5:
            return None
        
        # Try various structured patterns
        # For rank r, we can try to assign each term to cover multiple products
        
        # One approach: Use the fact that some terms can be shared
        # e.g., if we have terms that contribute to multiple (i,j,k) triples
        
        # For now, try a few specific structured patterns
        patterns = []
        
        # Pattern 1: Group by i (2 groups of 4x5)
        # Pattern 2: Group by j (4 groups of 2x5)
        # Pattern 3: Group by k (5 groups of 2x4)
        
        # Try to find a pattern where each rank-1 term covers multiple (i,j,k)
        # This is equivalent to finding a covering design
        
        # For small ranks, brute force specific structures
        if r <= 35:
            # Try to find terms that each cover at least 2 (i,j,k) pairs
            # This requires the tensor to have structure allowing such sharing
            
            # For 2x4x5, the tensor has 40 non-zero entries
            # If we can find r terms where each covers ~40/r entries, we might succeed
            
            # This is a set cover problem, which is NP-hard
            # For small r, we can try random set covers
            
            pass
        
        return None

    # Try ranks from 39 down to 30
    for r in range(rank_schoolbook - 1, 25, -1):
        if time.time() > deadline - 5:
            break
        
        # Try random search first (fast)
        result = try_random_rank(r, max_attempts=50)
        if result is not None:
            U, V, W = result
            best_solution = {'U': U, 'V': V, 'W': W}
            best_rank = r
            continue  # Try to find even lower
        
        # If random didn't work, try structured (slower)
        if time.time() < deadline - 10:
            result = structured_search(r)
            if result is not None:
                U, V, W = result
                best_solution = {'U': U, 'V': V, 'W': W}
                best_rank = r

    # If we haven't found anything better, return schoolbook
    return {
        "U": best_solution['U'].tolist(),
        "V": best_solution['V'].tolist(),
        "W": best_solution['W'].tolist()
    }
