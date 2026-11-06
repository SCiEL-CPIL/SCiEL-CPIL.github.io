import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing ones.
    Uses a NumPy boolean array initialized to True, then converts to string.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Create a NumPy array of True values (equivalent to '1' bits)
    bits = np.ones(N, dtype=bool)
    
    # Check deadline (this will almost never trigger, but satisfies the directive)
    if time.time() > deadline:
        # Fallback to a valid, high-scoring string if we somehow hit the limit
        # (In practice, the above operations take microseconds)
        return "1" * N

    # Convert boolean array to a string of '1's and '0's
    # np.where converts True to '1' and False to '0'
    binary_string = ''.join(np.where(bits, '1', '0'))
    
    return binary_string
# EVOLVE-BLOCK-END
