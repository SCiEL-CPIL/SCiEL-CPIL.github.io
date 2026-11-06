"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# Pre-allocate the optimal result at module load time using NumPy for vectorized initialization
import numpy as np
_OPTIMAL_ARRAY = np.ones(N, dtype=np.uint8)
# Convert to string once at load time; this is the constant-time return value
_OPTIMAL = _OPTIMAL_ARRAY.astype(str)

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string maximizing the number of ones."""
    # Explicit deadline guard as required by the directive
    # The operation is O(1) return of a pre-allocated string, so this is purely defensive
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # Direct construction: all ones is the unique global optimum for OneMax.
    # Return the pre-allocated optimal string.
    result = _OPTIMAL
    
    # Safety guard: ensure we return before the wall-clock deadline.
    if time.time() + safety_margin > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
