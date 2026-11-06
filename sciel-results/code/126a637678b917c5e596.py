"""
OneMax (100 bits) direct binary construction.
Returns the optimal solution: a string of 100 '1's.
Includes a deadline guard for safety.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # The OneMax objective is to maximize the number of 1s in a binary string of length N.
    # The optimal solution is a string of all 1s, which yields a score of N.
    # This is a direct analytical solution, so no iterative search is needed.
    
    # Construct the optimal string directly.
    result = "1" * N
    
    # Check deadline (safety margin)
    # In this trivial case, execution is instantaneous, but we include the guard
    # as required by the directive. If time were to exceed the limit (impossible here),
    # we would return the best result found so far.
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 1.0:
        # Return the current best result (which is optimal)
        return result
        
    return result
# EVOLVE-BLOCK-END
