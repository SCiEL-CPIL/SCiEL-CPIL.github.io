import numpy as np

N = 100

# Pre-allocate the optimal solution as a NumPy array at module load time.
# This satisfies the directive to shift initialization to import time.
_OPTIMAL_ARR = np.ones(N, dtype=np.int8)

def construct():
    """
    Return a binary string of length N maximizing the number of ones.
    
    Uses a pre-allocated NumPy array of ones. Converts to string upon invocation.
    This ensures constant-time execution with minimal branching logic.
    """
    # Convert the NumPy array of integers to a string of '1's
    # astype(str) converts each element to its string representation
    # join concatenates them
    return "".join(_OPTIMAL_ARR.astype(str))
