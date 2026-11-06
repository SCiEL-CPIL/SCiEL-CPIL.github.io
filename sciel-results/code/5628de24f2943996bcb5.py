"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is a string of all ones.
    """
    # Deadline guard (safety margin, though operation is O(1))
    start_time = time.time()
    deadline = start_time + 119.0  # 1 second safety margin
    
    # The optimal solution is all ones
    best_solution = "1" * N
    
    # Check if we are still within the time limit (always true here)
    if time.time() > deadline:
        # If somehow we were slow (impossible for string multiplication), return best found
        return best_solution
        
    return best_solution
# EVOLVE-BLOCK-END
