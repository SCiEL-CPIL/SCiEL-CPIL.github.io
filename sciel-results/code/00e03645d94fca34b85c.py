import numpy as np
import time

# Precompute the evaluation grid
# We use a dense grid to approximate the sup-norm.
# For N=24 coefficients, the polynomial degree is 23.
# A grid size of 4096 is typically sufficient for high accuracy in local search.
N_GRID = 4096
THETAS = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
# Z[i, j] = exp(1j * j * THETAS[i])
# Shape: (N_GRID, 24)
J = np.arange(24)
Z = np.exp(1j * np.outer(THETAS, J))

def get_score(c):
    """
    Compute the sup-norm of P(z) = sum c_k z^k on the grid.
    c: numpy array of shape (24,) with values +/- 1.
    """
    # Values of P(z) at all grid points
    # c is (24,), Z.T is (24, N_GRID). Result is (N_GRID,)
    vals = Z.T @ c
    return np.max(np.abs(vals))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    best_c = None
    best_score = float('inf')
    
    # Helper to update best
    def update_best(c, score):
        nonlocal best_c, best_score
        if score < best_score:
            best_score = score
            best_c = c.copy()
            
    # 1. Greedy Construction
    # Start with empty polynomial, add coefficients one by one.
    # This provides a good initial candidate.
    c_greedy = np.zeros(24, dtype=np.int8)
    for k in range(24):
        # Try +1
        c_greedy[k] = 1
        score_plus = get_score(c_greedy)
        # Try -1
        c_greedy[k] = -1
        score_minus = get_score(c_greedy)
        
        if score_plus < score_minus:
            c_greedy[k] = 1
            curr_score = score_plus
        else:
            c_greedy[k] = -1
            curr_score = score_minus
            
    update_best(c_greedy, curr_score)
    
    # 2. Local Search with Random Restarts
    # We will perform multiple random restarts.
    # Each restart: initialize random c, then improve via local search.
    # Local search: flip one bit if it improves the score.
    
    num_restarts = 0
    while time.time() < deadline:
        # Initialize random candidate
        c = np.random.choice([-1, 1], size=24).astype(np.int8)
        curr_score = get_score(c)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(24):
                # Try flipping bit i
                old_val = c[i]
                c[i] = -old_val
                new_score = get_score(c)
                
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                else:
                    # Revert
                    c[i] = old_val
                    
        update_best(c, curr_score)
        num_restarts += 1
        
        # Occasionally, try a "noisy" local search or simulated annealing step
        # if we are stuck at a local minimum.
        # For simplicity and speed, random restarts are often sufficient for n=24.
        
        # If we have done many restarts, we might increase the "temperature" 
        # or just keep going with random restarts.
        # Given the small dimension, 1000-5000 restarts should be feasible within time.
        
    # Final refinement on the best candidate found
    # We can do a more intensive local search on the best candidate found so far.
    if best_c is not None:
        c = best_c.copy()
        curr_score = best_score
        # Try multiple passes of local search
        for _ in range(10):
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(24):
                    old_val = c[i]
                    c[i] = -old_val
                    new_score = get_score(c)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                    else:
                        c[i] = old_val
            update_best(c, curr_score)
            
    return list(best_c)

# To ensure the function is ready, we can do a quick self-test if run directly, 
# but the task requires just the construct function.
# However, we must ensure Z is computed correctly.
# Z shape: (4096, 24). Z.T @ c -> (4096,). Correct.

