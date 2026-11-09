import numpy as np
# EVOLVE-BLOCK-START
def construct():
    """Construct a 100-bit binary string maximizing ones using NumPy vectorization."""
    # Initialize a NumPy array of uint8 with 1s
    bits = np.ones(100, dtype=np.uint8)
    # Format each element as the character '1'
    chars = np.char.mod('1', bits)
    # Join into a single string
    return ''.join(chars)
# EVOLVE-BLOCK-END
