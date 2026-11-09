"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The objective is OneMax: maximize the count of '1' characters in a
    binary string of length N. The optimal solution is all ones.
    
    This function returns the analytical optimum directly.
    """
    # Explicit deadline guard as per directive.
    # The operation is O(N) for string creation and O(1) for logic, 
    # so it completes in microseconds. The guard is for safety compliance.
    start_time = time.time()
    # Use a margin well below the 120s/150s kill limit.
    deadline = start_time + 10.0
    
    # The optimal solution for OneMax is a string of all '1's
    best_candidate = "1" * N
    
    # Defensive check: ensure length is exactly N
    # (Redundant for "1"*N but follows robustness pattern from inspirations)
    if len(best_candidate) != N:
        best_candidate = best_candidate[:N]
        
    # Check deadline (should never trigger for this O(1) logic + O(N) alloc)
    if time.time() > deadline:
        # Fallback to a valid string if somehow we hit a timeout (impossible here)
        # Returning all ones is still the best attempt
        return best_candidate
        
    return best_candidate
# EVOLVE-BLOCK-END
