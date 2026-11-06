import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string with all bits set to 1.
    Uses a NumPy boolean array initialized to True to leverage vectorized 
    memory representation, then converts it to a string of '1's.
    Includes a deadline guard to ensure completion within the time limit.
    """
    t0 = time.perf_counter()
    limit = t0 + 145.0  # Safety margin for 150s limit

    # Initialize the binary string as a NumPy boolean array of all True
    # This represents a string of all ones
    bits = np.ones(N, dtype=bool)

    # Convert the boolean array to a string of '1's and '0's
    # np.where is vectorized and efficient for this transformation
    char_array = np.where(bits, '1', '0')
    result = ''.join(char_array)

    # Check deadline (should be instantaneous, but guard is required)
    if time.perf_counter() > limit:
        # If for some reason we are too slow, return the optimal string directly
        return "1" * N

    return result
# EVOLVE-BLOCK-END
