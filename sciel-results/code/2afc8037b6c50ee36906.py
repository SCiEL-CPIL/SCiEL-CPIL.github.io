"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is all ones. We return it directly.
    Uses NumPy for vectorized construction as per directive.
    Includes a deadline guard as required.
    """
    t0 = time.monotonic()
    deadline = t0 + 145.0  # Safety margin before 150s limit
    
    # The OneMax problem with 100 bits has a clear global optimum:
    # all bits set to 1, giving a score of 100.
    # No search is needed; we can construct the optimal solution directly.
    
    # Vectorized construction using NumPy
    # Initialize array of 1s
    arr = np.ones(N, dtype=np.uint8)
    # Convert to string representation of each element
    # np.char.mod('1', arr) formats each integer 1 as the string '1'
    str_arr = np.char.mod('1', arr)
    # Join into a single string
    result = "".join(str_arr)
    
    # Verify we're within deadline
    if time.monotonic() < deadline:
        return result
    else:
        # Fallback: return what we have (which is already optimal)
        return result
# EVOLVE-BLOCK-END
