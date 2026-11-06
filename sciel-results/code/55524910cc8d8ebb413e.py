"""
Rewritten program for OneMax (100 bits).
Uses a direct analytical solution since the objective is linear and separable.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # The OneMax objective is to maximize the number of 1s.
    # The optimal solution is a string of all 1s.
    # Parameterization: Let x be a vector of length N.
    # Objective: sum(x)
    # Gradient: 1 for all x_i.
    # Since x_i must be in {0, 1}, the maximum is at x_i = 1.
    
    # Construct the optimal string directly.
    result = "1" * N
    
    # Check deadline (safety margin)
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 1.0:
        # Return the best result found so far (which is the optimal solution)
        return result
        
    return result
# EVOLVE-BLOCK-END
