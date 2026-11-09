"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # The task is OneMax with 100 bits: maximize the number of ones.
    # The optimal solution is all ones.
    # We use a NumPy boolean array initialized to True, then convert to string.
    # This leverages vectorized memory allocation.
    
    # Deadline guard: track elapsed time with a safety margin
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, but we'll be done in microseconds
    
    # Create a NumPy boolean array of length N, initialized to True (1)
    arr = [True] * N
    
    # Convert to binary string: True -> '1', False -> '0'
    result = ''.join('1' if b else '0' for b in arr)
    
    # Safety margin check (will always pass for this simple construction)
    if time.time() > deadline - 5:
        # Fallback: return the best result found (which is the optimal)
        return result
    
    return result
# EVOLVE-BLOCK-END
