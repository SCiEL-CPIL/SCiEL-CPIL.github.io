"""OneMax direct binary construction (100 bits)."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns the globally optimal string of all ones.
    Uses NumPy vectorized construction as per directive.
    """
    t0 = time.monotonic()
    limit = 145.0
    elapsed = time.monotonic() - t0
    if elapsed > limit:
        # Fallback: still return the optimal string
        return "1" * N
    
    try:
        # Strategy: Represent as NumPy uint8 array, then cast to string
        arr = np.ones(N, dtype=np.uint8)
        # Using astype(str) is the robust NumPy way to get string representations.
        # np.char.mod('1', arr) might be intended but is non-standard/ambiguous.
        # We use the vectorized allocation and C-level conversion pathway.
        return ''.join(arr.astype(str))
    except Exception:
        return "1" * N
# EVOLVE-BLOCK-END
