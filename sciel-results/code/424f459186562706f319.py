"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    This is the global optimum for the OneMax objective.
    
    Uses integer bitwise operations to generate the all-ones state,
    then converts to binary string. Includes explicit deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit
    
    # Direct construction of the global optimum: a string of N ones
    # This is the most efficient way to generate the optimal OneMax solution
    binary_string = "1" * N
    
    # Explicit deadline guard
    if time.time() > deadline:
        # Fallback: return a valid string if somehow we hit the deadline
        # In this constant-time case, this branch is effectively unreachable
        # but ensures we return *something* valid if the environment is extremely slow
        return "1" * N
        
    return binary_string
# EVOLVE-BLOCK-END
