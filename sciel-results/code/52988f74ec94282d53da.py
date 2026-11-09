"""
Program for OneMax direct binary construction (100 bits).
Goal: Maximize the number of ones in the binary string.
Optimal solution: A string of 100 '1's.
"""

N = 100

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a binary string of length N with maximum number of ones.
    Returns a string of N '1's.
    
    Includes a deadline guard to ensure compliance with time limits,
    although the optimal solution is constructed in O(N) time.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin below 120s limit
    
    # The optimal solution for OneMax is all ones.
    # We construct it directly.
    result = "1" * N
    
    # Check deadline (should never trigger for N=100)
    if time.time() - start_time > deadline:
        # Return best found (which is optimal) if somehow we hit a delay
        return result
        
    return result
# EVOLVE-BLOCK-END
