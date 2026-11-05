import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The global optimum for OneMax is a string of all 1s.
    We directly construct this solution in O(N) time.
    Includes an explicit deadline guard as required by the compute budget directive,
    although the operation is instantaneous.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Direct construction of the optimal solution: all bits set to 1
    result = "1" * N
    
    # Verify we are within the time limit (should always be true for N=100)
    if time.time() <= deadline:
        return result
    else:
        # Fallback: return the constructed result even if deadline exceeded 
        # (should never happen, but ensures a valid return type)
        return result
# EVOLVE-BLOCK-END
