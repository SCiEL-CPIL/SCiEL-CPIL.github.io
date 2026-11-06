"""Direct OneMax construction: return the optimal 100-bit all-ones string."""

import time

N = 100
TIME_LIMIT = 120.0  # Seconds
SAFETY_MARGIN = 1.0 # Seconds

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with the maximum possible number of ones."""
    start_time = time.time()
    
    # The OneMax objective is maximized by setting every bit to 1.
    # Construct the optimal solution in O(n) time with no search required.
    # Add explicit deadline guard as per directive, though computation is trivial.
    if time.time() - start_time > (TIME_LIMIT - SAFETY_MARGIN):
        return "0" * N  # Fallback, though unreachable in practice

    result = "1" * N
    
    # Ensure we return before the hard limit
    if time.time() - start_time > (TIME_LIMIT - SAFETY_MARGIN):
        return result
        
    return result
# EVOLVE-BLOCK-END
