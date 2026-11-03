"""
Rewritten program using analytical derivation.
The task is to minimize the maximum overlap of h with shifted 1-h.
Let C_k = sum_i h_i (1 - h_{i+k}).
Sum_k C_k = sum_k sum_i (h_i - h_i h_{i+k}) = sum_i (N h_i - h_i sum_j h_j) = sum_i h_i (N - S).
Given S = N/2, Sum_k C_k = (N/2)(N/2) = N^2/4.
Average C_k = N/4.
Max C_k >= Average C_k = N/4.
The uniform solution h_i = 0.5 yields C_k = N/4 for all k.
Thus, h_i = 0.5 is the global optimum.
"""

import numpy as np

N = 100

def construct():
    """
    Return the optimal step heights.
    
    The analytical solution is the uniform distribution h_i = 0.5 for all i.
    This achieves the theoretical lower bound for the maximum overlap.
    """
    # The optimal solution is a constant vector of 0.5
    # This satisfies:
    # 1. h_i in [0, 1]
    # 2. sum(h_i) = N * 0.5 = N/2 = 50
    # 3. Minimizes max overlap (achieves the lower bound N/4)
    
    h = np.full(N, 0.5)
    
    # Return as a list to ensure compatibility with "sequence" interface
    return list(h)
