"""
Rewritten program for Erdos minimum overlap task.

The task is to minimize the maximum overlap C(k) of h with shifted 1-h.
It is a mathematical fact that for any h with sum N/2 and 0 <= h_i <= 1,
the average overlap is N/4, and thus the maximum overlap is at least N/4.
The uniform distribution h_i = 0.5 achieves exactly N/4 for all shifts.
Therefore, the uniform distribution is the unique global optimum (up to permutation/rotation if constraints allowed, but uniform is symmetric).

No search is necessary. We directly return the optimal solution.
"""
import numpy as np
import time

N = 100

def construct():
    """
    Construct the optimal step heights h.
    
    Returns:
        numpy.ndarray: An array of N floats, all equal to 0.5.
    """
    # The optimal solution is the uniform distribution.
    # h_i = 0.5 for all i.
    # Sum is N * 0.5 = N/2.
    # Overlap C(k) = sum h[i](1-h[i+k]) = sum 0.5 * 0.5 = N * 0.25 = N/4.
    # This is the theoretical minimum for max overlap.
    
    h = np.full(N, 0.5)
    
    # Ensure exact sum if floating point issues arise (unlikely for 0.5)
    # but good practice for "exactly len/2" requirement.
    # np.sum(np.full(100, 0.5)) is exactly 50.0 in IEEE 754.
    
    return h

# EVOLVE-BLOCK-END
