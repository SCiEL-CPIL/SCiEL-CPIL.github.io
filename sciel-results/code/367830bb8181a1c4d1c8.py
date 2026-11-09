"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones: "1" * 100.
    
    We return this directly. A simple deadline guard is included for safety,
    though the computation is instantaneous.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well before 150s limit
    
    # The optimal solution for OneMax is all ones
    best = "1" * N
    
    # Check deadline (should never trigger, but included as safety guard)
    if time.time() > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END
