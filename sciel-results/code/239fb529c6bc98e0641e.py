"""OneMax direct binary construction (100 bits)."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is a string of 100 '1's.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Model the output as a NumPy boolean array initialized to True.
    # This represents 100 ones.
    bits = np.ones(N, dtype=bool)
    
    # Convert the boolean array to a binary string.
    # astype('U1') converts True to '1' and False to '0' in a string-like array,
    # but we need a single string. 
    # A more robust way to convert numpy bool array to string of '1's and '0's:
    # np.array2string(bits, separator='') gives something like "True True ..."
    # Instead, we can map True to '1' and False to '0'.
    
    # Since we know all are True, we could just return '1'*N.
    # But following the directive to use a distinct numerical representation pathway:
    
    # Method: Convert to integers (1 for True, 0 for False) and then to string.
    int_bits = bits.astype(np.uint8)
    
    # Check deadline (unlikely to be hit)
    if time.time() > deadline:
        # Fallback to simple string construction if somehow delayed
        return "1" * N

    # Convert array of 0s and 1s to string
    # np.array2string(int_bits, separator='') produces "111...1"
    result_str = np.array2string(int_bits, separator='')
    
    # Ensure it's exactly length N and contains only 0/1
    if len(result_str) != N:
        return "1" * N
        
    return result_str
# EVOLVE-BLOCK-END
