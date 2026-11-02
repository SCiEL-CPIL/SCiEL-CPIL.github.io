"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback (Rank 50)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # Pre-compute index mappings for faster verification
    # Valid triples: (i, j, k) -> (alpha, beta, gamma)
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_triples.append((alpha, beta, gamma))
                
    # Invalid triples: (i, j, j', k) with j != j'
    # Note: The tensor is zero if the middle index j in U doesn't match j in V.
    # U index: i*m+j. V index: j'*p+k.
    # If j != j', the term is 0.
    invalid_triples = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp: continue
                for k in range(p):
                    alpha = i * m + j
                    beta = jp * p + k
                    gamma = k * n + i
                    invalid_triples.append((alpha, beta, gamma))
    
    def verify(U, V, W):
        """
        U, V, W are numpy arrays of shape (rows, R).
        Checks if sum_r U[alpha,r] V[beta,r] W[gamma,r] == 1 for valid, 0 for invalid.
        """
        # Vectorized check for valid triples
        # For each valid triple (a,b,g), compute dot(U[a] * V[b], W[g])
        # We can do this by iterating, but vectorization helps if R is large.
        # Given small sizes (10, 25, 10), simple loops with numpy dot might be fine,
        # but let's try to be efficient.
        
        # Check valid
        for a, b, g in valid_triples:
            # val = np.sum(U[a] * V[b] * W[g])
            # Using dot for speed
            val = np.dot(U[a] * V[b], W[g])
            if val != 1:
                return False
        
        # Check invalid
        for a, b, g in invalid_triples:
            val = np.dot(U[a] * V[b], W[g])
            if val != 0:
                return False
                
        return True

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # We know the tensor rank is likely 43 or 44 for 2x5x5.
    # Schoolbook is 50.
    # Let's try to find a decomposition with rank < 50.
    # We will search from low ranks upwards.
    # However, random search for exact integer solutions is hard.
    # A better strategy for "exact tensor decomposition" with small integers:
    # Start with the schoolbook decomposition and try to merge columns?
    # Or use a structured random search.
    
    # Given the difficulty of finding exact integer solutions via pure random search in high dimensions,
    # and the time limit, we should focus on a heuristic that might find a lower rank.
    # One known technique is to use the structure of the matrix multiplication tensor.
    
    # Let's try a specific structured search for rank 43, 44, 45, 46, 47, 48, 49.
    # We will use random initialization with small integers and check.
    # To increase success rate, we can use a "local search" or "annealing" style,
    # but exactness makes it tricky.
    
    # Alternative: Use the fact that we can permute and scale.
    # Let's just do a robust random search with more attempts for lower ranks.
    
    max_rank_to_try = rank_school - 1
    min_rank_to_try = 40 # Lower bound is likely around 43
    
    # We will iterate R from min_rank_to_try to max_rank_to_try
    # For each R, we try a number of random restarts.
    
    # To make random search more effective, we can bias towards solutions that are "close" to schoolbook.
    # But schoolbook has orthogonal columns.
    
    # Let's try a different approach:
    # Start with schoolbook (R=50).
    # Try to find linear dependencies among the columns of U, V, W that allow merging.
    # This is equivalent to finding a lower rank decomposition.
    
    # Given the constraints, a simple random search might be too slow to find 43.
    # However, we can try a few specific structured candidates.
    
    # For now, we will stick to random search but optimize the verification.
    # We can pre-compute the "target" tensor and use a faster check?
    # No, the check is already O(N_valid * R + N_invalid * R).
    
    # Let's increase the number of attempts for each rank if time permits.
    
    for R in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Number of attempts depends on remaining time and rank
        # Lower ranks are harder, so we might want to spend more time on them?
        # Actually, if a low rank solution exists, it's a "needle in a haystack".
        # But higher ranks (e.g. 49) might be easier to find?
        # Actually, we want the MINIMAL rank. So we must check 43 before 44, etc.
        
        # Let's set a time budget per rank.
        rank_start = time.time()
        max_time_per_rank = 10 # 10 seconds per rank
        attempts = 0
        found = False
        
        while time.time() - rank_start < max_time_per_rank and not found:
            attempts += 1
            
            # Random initialization
            # Use small integers to keep numbers manageable
            # -1, 0, 1 are good candidates
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            if verify(U, V, W):
                found = True
                best_rank = R
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                # If we found a solution for R, we can stop checking this R
                # and move to R-1? No, we are iterating upwards.
                # Wait, if we found 43, we should stop entirely?
                # No, we want the MINIMAL rank.
                # If we are iterating from 40 upwards, and we find 43,
                # we should check 42, 41, 40?
                # But we are iterating 40, 41, 42, 43...
                # So if we find 43, we have already checked 40, 41, 42 and failed.
                # So 43 is the best found so far.
                # We can break out of the R loop?
                # Yes, because we are looking for the minimum, and we go in increasing order.
                # If we find a solution for R, it is the minimal one found.
                # We can stop searching for higher ranks.
                break
        
        if found:
            # We found a solution for this R. Since we iterate R increasing,
            # this is the minimal rank found.
            # We can stop the outer loop.
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
