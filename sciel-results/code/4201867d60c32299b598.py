import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global optimum for the OneMax problem is a string of all '1's.
    This solution returns the optimal string directly, ensuring maximum score
    with minimal execution time.
    """
    # Explicit deadline guard (safety margin)
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Return the theoretical maximum: 100 ones
    # This is O(1) and cannot be improved.
    if time.time() < deadline:
        return "1" * N
    else:
        # Fallback (should never happen)
        return "1" * N
# EVOLVE-BLOCK-END
