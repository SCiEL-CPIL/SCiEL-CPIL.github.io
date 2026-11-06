"""
Rewritten program for OneMax (100 bits).
Directly constructs the global optimal solution: a string of 100 '1's.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # The OneMax objective is to maximize the number of 1s in a binary string.
    # The global optimum is a string of all 1s.
    # We construct this directly.
    
    # Construct the optimal string
    result = "1" * N
    
    # Explicit deadline guard as per directive
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 1.0:
        # In this deterministic case, we never exceed the deadline,
        # but we return the best result found so far (the optimal solution).
        return result
        
    return result
# EVOLVE-BLOCK-END
