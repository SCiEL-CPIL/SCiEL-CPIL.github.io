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
    # Task contract specifies 120s limit. We use a 110s threshold as a safety margin.
    # Since construction is O(N) and instantaneous, this guard will not trigger,
    # but it satisfies the requirement to track elapsed time and handle deadlines.
    if time.time() - start_time > 110:
        return result
        
    return result
# EVOLVE-BLOCK-END
