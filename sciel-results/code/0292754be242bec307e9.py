import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Initialize a NumPy boolean array of length N with all True values
    arr = np.full(N, True, dtype=bool)
    # Convert the boolean array to a string of '1's
    # astype('U1') converts True to '1' and False to '0'
    return ''.join(arr.astype('U1'))
# EVOLVE-BLOCK-END
