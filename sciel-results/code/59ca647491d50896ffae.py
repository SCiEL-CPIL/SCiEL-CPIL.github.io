"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The global optimum for OneMax is a string of all 1s.
    """
    import time
    start_time = time.time()
    
    # Direct construction of the global optimum
    result = "1" * N
    
    # Explicit deadline guard
    # If we were doing iterative search, we would check time here.
    # For this O(N) construction, we simply return.
    if time.time() - start_time > 140:
        return result
        
    return result
# EVOLVE-BLOCK-END
