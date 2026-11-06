import time

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Since the objective is to maximize the number of ones,
    the optimal solution is all ones.
    """
    start_time = time.time()
    
    # The OneMax objective is to maximize the number of 1s.
    # The optimal solution is a string of all 1s.
    # This is computed in O(N) time, which is negligible for N=100.
    result = "1" * N
    
    # Safety guard: Check if we have exceeded the deadline (unlikely, but required by directive)
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 1.0:
        # In this trivial case, we should never hit this, but it's good practice.
        # If we were doing search, we would return best_so_far.
        pass
        
    return result
# EVOLVE-BLOCK-END
