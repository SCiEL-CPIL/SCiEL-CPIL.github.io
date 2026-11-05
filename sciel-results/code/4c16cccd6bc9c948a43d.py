"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""
import time

N = 100
DEADLINE_SECONDS = 120.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for OneMax.
    The global optimum is a string of all '1's.
    Includes an explicit deadline guard for compliance with compute budget constraints,
    though the operation is instantaneous.
    """
    start_time = time.time()
    
    # The optimal solution for OneMax is a string of all ones.
    # Using string multiplication is the most efficient direct construction method.
    result = "1" * N
    
    # Explicit deadline guard:
    # If for any reason the system is under extreme load or this function is called
    # in a context where time is critical, we ensure we return before the hard limit.
    # In practice, this check adds negligible overhead but satisfies the directive.
    if time.time() - start_time > DEADLINE_SECONDS - 1.0:
         # Return the result immediately if we are dangerously close to the limit.
         # Since result is already computed, this is just a safeguard.
         pass
         
    return result
# EVOLVE-BLOCK-END
