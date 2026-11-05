"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    For OneMax, the global optimum is all 1s. We can directly construct
    this solution in O(N) time, which is well within any time limit.
    """
    # Set a deadline guard for safety (though this task is trivially fast)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin under 120s limit
    
    # Direct construction of the optimal solution: all bits set to 1
    # This is the global optimum for OneMax with N=100
    result = "1" * N
    
    # Safety check: ensure we're within deadline (should always be true)
    if time.time() < deadline:
        return result
    else:
        # Fallback: return whatever we have (should never happen for this simple task)
        return result
# EVOLVE-BLOCK-END
