"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # The OneMax objective is simply the count of '1's in the binary string.
    # The maximum possible score for a string of length N is N, achieved by all '1's.
    # Since this is a direct construction task (not an evolutionary search),
    # we can directly return the optimal solution.
    
    # Deadline guard (safety margin before 150s limit)
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Directly construct the optimal solution using NumPy boolean array
    # Initialize an array of True booleans
    arr = np.ones(N, dtype=bool)
    
    # Convert boolean array to string of '1's
    # np.where(arr, '1', '0') creates an array of strings, which we join
    result = ''.join(np.where(arr, '1', '0'))
    
    # Safety check (will never trigger for O(N) construction)
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
