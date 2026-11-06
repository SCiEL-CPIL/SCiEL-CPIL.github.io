"""OneMax direct binary construction (100 bits)."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # The optimal solution for OneMax is all ones.
    # Use NumPy to create an array of ones and convert to a string.
    bits = np.ones(N, dtype=int)
    return ''.join(str(b) for b in bits)
# EVOLVE-BLOCK-END
