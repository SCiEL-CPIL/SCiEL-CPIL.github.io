"""OneMax direct binary construction (100 bits)."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is a string of all '1's.
    """
    # Deadline guard: track elapsed time and return best result with safety margin.
    # Since the optimal solution is trivially known, we can return it immediately.
    # This is the best possible result: 100 ones.
    
    # Use NumPy boolean array initialized to True for vectorized construction
    bits = np.ones(N, dtype=bool)
    
    # Convert boolean array to string of '1's efficiently
    # Using a list comprehension with str() conversion
    return ''.join('1' if b else '0' for b in bits)
# EVOLVE-BLOCK-END
