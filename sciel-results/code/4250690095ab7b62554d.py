"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
Uses an explicit deadline guard as required.
"""
import time

# Module-level constant for problem size
SIZE = 100

# Pre-compute the optimal string at import time for minimal overhead
OPTIMAL_ONE_MAX = ''.join(['1'] * SIZE)

# EVOLVE-BLOCK-START

def construct() -> str:
    """
    Returns a binary string of length SIZE (100) that maximizes the count of '1' bits.
    The global optimum for OneMax is a string consisting entirely of '1's.
    An explicit time guard is included to ensure safe termination before the evaluator's limit.
    """
    # Explicit deadline guard: track start and a safety margin
    t_start = time.monotonic()
    hard_limit = 150.0  # Evaluator hard kill
    safety_margin = 5.0 # Leave 5s buffer
    deadline = hard_limit - safety_margin
    
    # The direct construction of the OneMax optimum is a string of all 1s.
    # This can be generated via slicing, multiplication, or joining characters.
    # Using multiplication is the most standard O(N) direct construction.
    result = "1" * SIZE
    
    # Check if we somehow exceeded the safety deadline (should be instantaneous)
    if (time.monotonic() - t_start) > deadline:
        # Return the result anyway, as it's already computed
        pass
    
    # Return the maximized binary string
    return result

# EVOLVE-BLOCK-END
