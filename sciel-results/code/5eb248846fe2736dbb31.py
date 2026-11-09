"""Program for OneMax direct binary construction (100 bits)."""
import time
import sys

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The direct, optimal construction is to set all bits to '1'.
    This achieves the maximum possible score of N (100 ones).
    """
    # Direct approach: all ones is the optimal solution for OneMax.
    # For N=100, "1" * 100 is the unique global optimum, yielding the maximum score.
    best_result = "1" * N
    
    # Explicit deadline guard: return best result if time limit is exceeded.
    # This is redundant for this O(N) operation but ensures compliance with constraints.
    if time.time() > 120.0:
        return best_result
        
    return best_result
# EVOLVE-BLOCK-END
