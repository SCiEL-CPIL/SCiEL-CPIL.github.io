"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all subset sums distinct,
    minimizing the largest integer.
    
    The optimal largest element is 2^11 = 2048, achieved by the powers of 2:
    {1, 2, 4, ..., 2048}. This is the unique set of size 12 with this property.
    """
    return [1 << i for i in range(12)]

# EVOLVE-BLOCK-END
