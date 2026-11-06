"""
Task: OneMax direct binary construction (100 bits)
Objective: Maximize the number of ones in a 100-bit binary string.
Solution: Return a string of 100 ones.
"""

N = 100

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a 100-bit binary string with all bits set to 1.
    This maximizes the OneMax objective function (100 ones).
    """
    # Explicit deadline guard as per directive
    start_time = time.time()
    deadline = time.time() + 145.0  # 145s absolute limit or 5s margin from start? 
    # The directive says "track elapsed time and return the best result found with a safety margin before the limit."
    # The limit is 150s. So deadline should be 145s from start.
    
    # Optimal solution: all ones
    result = "1" * N
    
    # Check if we are still within the deadline (always true for this O(N) op)
    if time.time() - start_time < 145.0:
        return result
    else:
        # Fallback (should never happen)
        return result
# EVOLVE-BLOCK-END
