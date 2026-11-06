"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction of the global optimum for OneMax(100).
    Returns a string of 100 '1's.
    """
    t_start = time.monotonic()
    deadline = t_start + 149.0
    
    # The optimal solution for OneMax(N) is a string of N ones.
    # This is an O(N) operation which is instantaneous for N=100.
    best_solution = "1" * N
    
    # Explicit deadline guard as per compute budget directive.
    if time.monotonic() < deadline:
        return best_solution
    
    # Fallback for safety (should not occur)
    return best_solution
# EVOLVE-BLOCK-END
