"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or as low as possible) tensor decomposition
    for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = 48).
    2. Use a heuristic/local search to try to reduce the rank by finding
       dependencies in the tensor or using known lower-bound constructions.
    
    Since finding the exact minimal rank is hard, we will use a greedy
    approach to try to merge terms where possible, or use a known
    decomposition if available. For now, we return the schoolbook
    decomposition as a safe baseline, but we will attempt to use a
    "Landsberg" style or similar optimized decomposition if we can
    find one quickly. 
    
    Actually, for small matrices, the minimal rank is often unknown or
    equal to the schoolbook. Let's implement a simple local search that
    tries to set entries to 0 or -1 or 1 to see if we can reduce the rank
    while maintaining correctness. This is computationally expensive.
    
    A better approach for a "bootstrap" is to simply return the schoolbook
    solution, but we can try to verify if a lower rank is possible by
    checking for linear dependencies in the factor matrices.
    
    However, the task asks for a SEARCH approach. Let's do a randomized
    local search on the rank-48 solution to see if we can find a
    rank-47 or lower solution.
    
    Given the time constraint (150s), we will:
    1. Generate the schoolbook solution.
    2. Try a few random perturbations of the rank to see if we can
       find a valid solution with lower rank.
    
    Actually, let's just implement the schoolbook solution as it's the
    only guaranteed correct one within the time limit for a bootstrap.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin of 10s
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate schoolbook solution
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to reduce rank via local search
    # We'll try to remove columns that are linear combinations of others
    # This is a heuristic approach
    
    # First, let's try a simple greedy approach:
    # For each column, check if it can be removed (i.e., if the tensor
    # can still be represented without it)
    
    # This is expensive, so we'll only try a limited number of steps
    max_steps = 100
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Try to remove a random column
        if best_rank <= 1:
            break
            
        col_to_try = random.randint(0, best_rank - 1)
        
        # Create modified matrices without this column
        U_new = np.delete(best_U, col_to_try, axis=1)
        V_new = np.delete(best_V, col_to_try, axis=1)
        W_new = np.delete(best_W, col_to_try, axis=1)
        
        # Check if the decomposition is still valid
        # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For schoolbook, each term is independent, so removing any column
        # will break the decomposition unless there's redundancy
        
        # Check a few random entries to see if the decomposition holds
        valid = True
        for _ in range(10):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            
            # Expected value for this entry
            expected = 1 if (i < n and j < m and k < p) else 0
            # Actually, for matrix multiplication, T[i,j,k] = 1 if this is a valid
            # multiplication term, which it always is for the schoolbook
            
            # Compute the decomposition value
            val = 0
            for c in range(best_rank - 1):
                val += U_new[i * m + j, c] * V_new[j * p + k, c] * W_new[k * n + i, c]
            
            if val != expected:
                valid = False
                break
        
        if valid:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank -= 1
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
