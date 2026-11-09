"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The OneMax objective is to maximize the count of '1's in a binary string of length N.
    The global optimum is a string consisting entirely of '1's.
    This solution returns the global optimum directly using Python's arbitrary-precision integers.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # The optimal solution is all ones.
    # Represent as an integer: (1 << N) - 1 gives N bits of 1s.
    val = (1 << N) - 1
    optimal_string = format(val, f'0{N}b')
    
    # Explicit deadline guard
    if time.time() < deadline:
        return optimal_string
    else:
        # Fallback: return the best result found (which is optimal)
        return optimal_string
# EVOLVE-BLOCK-END
