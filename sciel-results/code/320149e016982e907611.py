"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard to ensure we finish well within the 150s limit
    t0 = time.perf_counter()
    limit = t0 + 145.0
    
    # OneMax objective is maximized by a string of all ones.
    # Construct the optimal solution directly using a NumPy boolean array.
    # Initialize an array of True values (representing 1s).
    bits = np.ones(N, dtype=bool)
    
    # Convert the boolean array to a string of '1's.
    # np.char.mod is a vectorized way to format, but simple join or decode is often faster for small N.
    # However, to strictly follow the "array-based construction" hint:
    # We can use a lookup or direct string conversion.
    # Since we need a string of '1's:
    optimal_str = ''.join(['1' if b else '0' for b in bits])
    
    # Safety check: if we somehow exceed the time (impossible for this op), return anyway
    if time.perf_counter() > limit:
        return optimal_str
        
    return optimal_str
# EVOLVE-BLOCK-END
