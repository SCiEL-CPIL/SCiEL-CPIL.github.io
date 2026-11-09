"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE = 150.0
SAFETY_MARGIN = 1.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is a string of all '1's.
    Since the objective is deterministic and known (maximize count of '1's),
    we directly return the global optimum.
    """
    start_time = time.time()
    
    # Check if we're close to the deadline
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        return '1' * N
    
    # Use Python's arbitrary-precision integer to represent 100 ones
    # (1 << 100) - 1 creates a 100-bit integer with all bits set to 1
    val = (1 << N) - 1
    # Convert to binary string with zero-padding to ensure exactly N bits
    result = format(val, f'0{N}b')
    
    return result
# EVOLVE-BLOCK-END
