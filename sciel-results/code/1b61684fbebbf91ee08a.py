"""Program for OneMax direct binary construction (100 bits)."""
import time

N = 100
START_TIME = time.monotonic()
DEADLINE = 115.0  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    OneMax objective: maximize count of '1' bits in a length-N string.
    The unique global optimum is all ones. We return it directly.
    A deadline guard ensures we return a valid result even under
    pathological timing conditions.
    """
    result = '1' * N
    
    # Explicit deadline guard: if somehow we've exceeded the safety window,
    # return immediately. In practice, construction is O(N) with negligible cost.
    if (time.monotonic() - START_TIME) > DEADLINE:
        return result
        
    return result
# EVOLVE-BLOCK-END
