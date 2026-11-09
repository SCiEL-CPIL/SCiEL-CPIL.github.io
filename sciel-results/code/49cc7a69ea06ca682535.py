"""OneMax 100-bit binary construction: maximize number of ones."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    Uses a NumPy boolean array pathway as per design constraints.
    """
    start_time = time.time()
    deadline = 145.0
    
    try:
        # Initialize NumPy boolean array of all True (1s)
        bits = np.ones(N, dtype=bool)
        
        # Convert to string array of '1's and '0's
        # np.where is efficient for this vectorized mapping
        char_array = np.where(bits, '1', '0')
        
        # Join into final string
        result = ''.join(char_array)
        
        return result
    except Exception:
        # Fallback to direct string multiplication if NumPy fails or times out
        return "1" * N
# EVOLVE-BLOCK-END
