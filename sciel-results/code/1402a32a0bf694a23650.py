"""Optimized OneMax constructor: direct binary construction maximizing ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is to maximize the count of '1' bits.
    The global optimum is a string of length N consisting entirely of '1's.
    This function directly constructs this optimal string with a deadline guard.
    
    Returns:
        str: A binary string of length 100 consisting entirely of '1's.
    """
    start_time = time.time()
    deadline = start_time + 119.0  # 1 second safety margin before 120s limit
    
    # The optimal solution for OneMax is all ones.
    # This is O(N) time and O(N) space, which is negligible for N=100.
    optimal_solution = "1" * N
    
    # Check deadline (though this operation is instantaneous)
    if time.time() > deadline:
        # In case of extreme slowness (unlikely), return a valid fallback
        # but "1"*N is the best possible, so we just return it.
        return optimal_solution
        
    return optimal_solution
# EVOLVE-BLOCK-END
