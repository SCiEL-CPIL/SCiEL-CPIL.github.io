"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Uses big-integer conversion to generate the optimal string of all ones.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Strategy: Construct the optimal 100-bit string by converting the integer 
    # 2**100 - 1 to a binary string.
    # This leverages Python's internal big-integer bit manipulation.
    
    # Calculate the integer value for 100 ones: 2^100 - 1
    val = (1 << N) - 1
    
    # Convert to binary string representation and strip the '0b' prefix
    # bin(val) returns '0b' followed by the binary digits
    binary_str = bin(val)[2:]
    
    # Defensive check: ensure the length is exactly N
    # For N=100, bin(2**100 - 1) is '0b' + 100 '1's, so [2:] gives 100 '1's
    if len(binary_str) != N:
        # Fallback to direct string repetition if something unexpected happens
        binary_str = "1" * N
        
    # Deadline guard
    if time.time() > deadline:
        # Return the best result found (which is already optimal)
        return binary_str

    return binary_str
# EVOLVE-BLOCK-END
