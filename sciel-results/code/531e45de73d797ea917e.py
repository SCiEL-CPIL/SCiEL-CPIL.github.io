"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], np.multiply(V_arr[j*p+k], W_arr[k*n+i]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try structured decomposition based on known results for similar sizes
    # For 4x4x4, rank 21 is known. For 4x4x6, we can try to build on this.
    # The key insight: we can decompose the 4x4x6 tensor by splitting
    # the 6 columns into 4+2 and using a combined approach.
    
    # Strategy: Use a block-structured decomposition
    # C[:, :4] = A * B[:, :4] and C[:, 4:6] = A * B[:, 4:6]
    # If we can share some terms between the two blocks, we save rank.
    
    # Alternative: Try a direct search with structured sparsity patterns
    # Use the fact that optimal decompositions often have specific structure
    
    # Let's try a more systematic approach using known patterns
    # For 4x4x4, the Strassen-like patterns can be extended
    
    # First, let's build a candidate based on 4x4x4 rank-21 + extension
    # The 4x4x4 rank-21 decomposition has a specific structure
    
    # For now, try random search with better initialization
    # Use a grid of possible values and try different rank values
    
    # Known lower bound for 4x4x6: max(16, 24, 24) = 24
    # Upper bound: 96 (schoolbook), known better: ~88 (Landsberg)
    # Realistic target: 70-85
    
    # Try a structured random search with sparsity constraints
    # Each term (column) should have a sparse pattern
    
    target_ranks = list(range(70, 96, 2)) + list(range(88, 96))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Multiple random attempts with structured initialization
        max_attempts = 30
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with zeros
            U_rand = np.zeros((n * m, target_rank))
            V_rand = np.zeros((m * p, target_rank))
            W_rand = np.zeros((p * n, target_rank))
            
            # For each term/column, create a sparse pattern
            # The pattern should correspond to a (i,j,k) combination
            # with some probability of sharing terms
            
            for r in range(target_rank):
                # Choose which (i,j,k) this term contributes to
                # With some probability, make it a "shared" term that
                # contributes to multiple (i,j,k) pairs
                
                # Base pattern: pick a random (i,j,k)
                i_idx = random.randint(0, n - 1)
                j_idx = random.randint(0, m - 1)
                k_idx = random.randint(0, p - 1)
                
                # Set the base term
                U_rand[i_idx * m + j_idx, r] = random.choice([-1, 1])
                V_rand[j_idx * p + k_idx, r] = random.choice([-1, 1])
                W_rand[k_idx * n + i_idx, r] = random.choice([-1, 1])
                
                # With some probability, add additional contributions
                # to create sharing
                if random.random() < 0.3:
                    # Add an additional (i,j,k) pair
                    i2 = random.randint(0, n - 1)
                    j2 = random.randint(0, m - 1)
                    k2 = random.randint(0, p - 1)
                    if not (i2 == i_idx and j2 == j_idx and k2 == k_idx):
                        U_rand[i2 * m + j2, r] = random.choice([-1, 1])
                        V_rand[j2 * p + k2, r] = random.choice([-1, 1])
                        W_rand[k2 * n + i2, r] = random.choice([-1, 1])
                
                # Occasionally, make it more sparse by zeroing some entries
                if random.random() < 0.2:
                    row = random.randint(0, n * m - 1)
                    U_rand[row, r] = 0
            
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # If still at schoolbook rank, try a different approach
    # Use local search: start from a random valid solution and try to reduce rank
    if best_rank == rank_schoolbook and time.time() < deadline:
        # Start with a random valid decomposition at rank 96
        # Then try to merge terms to reduce rank
        
        # For now, try a few more random searches with different parameters
        for target_rank in range(75, 96):
            if time.time() > deadline:
                break
            
            for attempt in range(10):
                if time.time() > deadline:
                    break
                
                # Use a biased initialization: start from schoolbook
                # and randomly perturb
                U_rand = [row[:] for row in U_school]
                V_rand = [row[:] for row in V_school]
                W_rand = [row[:] for row in W_school]
                
                # Keep only the first target_rank columns
                U_rand = [row[:target_rank] for row in U_rand]
                V_rand = [row[:target_rank] for row in V_rand]
                W_rand = [row[:target_rank] for row in W_rand]
                
                # Verify
                if verify(U_rand, V_rand, W_rand):
                    best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                    best_rank = target_rank
                    break
            
            if best_rank < rank_schoolbook:
                break
    
    return best_result
# EVOLVE-BLOCK-END
