import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global optimum for the OneMax problem is a string of all '1's.
    """
    start_time = time.time()
    # Safety margin: 10 seconds is well within the 120s/150s limit
    deadline = start_time + 10.0
    
    # Construct the optimal solution
    result = "1" * N
    
    # Explicit deadline guard as per directive
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
