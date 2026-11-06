import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global optimum for the OneMax problem is a string of all '1's.
    """
    # Explicit deadline guard as per directive, though computation is instantaneous
    start_time = time.time()
    result = "1" * N
    
    # Safety margin check (redundant for constant time, but required by directive structure)
    if time.time() - start_time > 119:
        return result
        
    return result
# EVOLVE-BLOCK-END
