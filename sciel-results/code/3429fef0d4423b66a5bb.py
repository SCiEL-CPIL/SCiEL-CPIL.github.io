"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The objective is to maximize the count of '1's in the string.
    The global optimum is a string of all '1's.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # The optimal solution for OneMax is deterministic: all bits are 1.
    # Constructing this string is O(N) and extremely fast.
    result = "1" * N
    
    # Check if we are still within the deadline (always true for this operation,
    # but included to satisfy the explicit deadline guard requirement).
    if time.time() > deadline:
        # This branch is effectively unreachable for N=100, 
        # but ensures robustness if N were to change significantly.
        pass
        
    return result
# EVOLVE-BLOCK-END
