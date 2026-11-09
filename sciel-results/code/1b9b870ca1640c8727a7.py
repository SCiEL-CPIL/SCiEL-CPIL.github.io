"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns the globally optimal string of all ones.
    """
    t0 = time.monotonic()
    limit = 145.0
    elapsed = time.monotonic() - t0
    if elapsed > limit:
        # Fallback: still return the optimal string
        return "1" * N
    return "1" * N
# EVOLVE-BLOCK-END
