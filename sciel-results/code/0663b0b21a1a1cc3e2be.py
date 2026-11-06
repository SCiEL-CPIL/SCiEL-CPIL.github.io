"""
OneMax (100 bits) direct binary construction.
Combines the direct construction strategy with pre-allocation for minimal overhead.
"""
import time
import numpy as np

N = 100
DEADLINE_SECONDS = 150.0

# Pre-allocate the optimal solution at module load time.
# This ensures construct() executes in constant time with zero branching logic
# and no string concatenation or allocation overhead at call time.
_OPTIMAL_STR = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns the pre-computed optimal solution (all ones).
    """
    start_time = time.time()
    
    # The OneMax objective is separable and linear: f(x) = sum_i x_i with x_i in {0,1}.
    # The unique global maximizer is x_i = 1 for all i.
    # We return the pre-allocated optimal string directly.
    
    result = _OPTIMAL_STR
    
    # Explicit deadline guard as per directive
    # In this deterministic case, we never exceed the deadline,
    # but we check to satisfy the requirement.
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 1.0:
        # Return the best result found so far (the optimal solution).
        return result
        
    return result
# EVOLVE-BLOCK-END
