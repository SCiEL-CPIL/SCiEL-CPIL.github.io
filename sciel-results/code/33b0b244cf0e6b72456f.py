"""Program for OneMax direct binary construction (100 bits)."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Initialize a NumPy boolean array of length N with all True values.
    # Each True represents a '1' in the binary string.
    bits = np.ones(N, dtype=bool)
    # Convert the boolean array to a string of '1's and '0's using vectorized where.
    # This maps True to '1' and False to '0', then joins the array elements.
    return ''.join(np.where(bits, '1', '0'))
# EVOLVE-BLOCK-END
