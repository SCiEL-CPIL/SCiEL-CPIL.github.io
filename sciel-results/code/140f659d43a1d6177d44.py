"""OneMax direct binary construction (100 bits)."""
import time

N = 100
DEADLINE_SECONDS = 145.0  # safety margin before the 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    For OneMax, the global optimum is all ones. This is a deterministic
    construction that directly returns the optimal solution.
    
    We include a deadline guard as required, though the computation is
    instantaneous.
    """
    start_time = time.time()
    
    # The optimal solution for OneMax is a string of N ones.
    # This is a direct construction - no search needed.
    result = "1" * N
    
    # Verify length
    assert len(result) == N, f"Expected length {N}, got {len(result)}"
    
    # Deadline guard (will never trigger for this trivial computation)
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        # Fallback: still return the best we have (which is optimal)
        return result
    
    return result
# EVOLVE-BLOCK-END
