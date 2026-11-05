import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is maximized by setting all bits to 1.
    This function returns a string of N '1's.
    """
    # Deadline guard (though not strictly necessary for O(N) string construction)
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, keep 5s margin
    
    # The optimal solution for OneMax is a string of all ones.
    # We construct this directly.
    result = "1" * N
    
    # Check deadline (should never trigger for this simple construction)
    if time.time() > deadline:
        # In case of unexpected slowness, return what we have (which is already optimal)
        return result
        
    return result
# EVOLVE-BLOCK-END
