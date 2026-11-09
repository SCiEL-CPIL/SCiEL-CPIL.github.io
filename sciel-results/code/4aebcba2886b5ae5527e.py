import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Create a NumPy boolean array of length N, initialized to True (1)
    arr = np.ones(N, dtype=bool)
    # View the boolean array as unsigned integers (0 and 1)
    # This is a zero-copy operation that reinterprets the memory
    uint_arr = arr.view(np.uint8)
    # Convert to single-character Unicode strings ('0' and '1')
    str_arr = uint_arr.astype('U1')
    # Join the array elements into a single string
    return ''.join(str_arr)
# EVOLVE-BLOCK-END
