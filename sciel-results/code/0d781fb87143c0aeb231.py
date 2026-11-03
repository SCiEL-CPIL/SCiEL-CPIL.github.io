"""
Optimized program for the Erdos minimum overlap task.

The task is to construct a sequence of step heights h of length N,
where each h_i is in [0, 1] and sum(h) = N/2, to minimize the
maximum overlap C(k) = sum_i h_i * (1 - h_{i+k}) over all shifts k.

Theoretical Analysis:
Let x_i = 2h_i - 1. Then x_i is in [-1, 1] and sum(x_i) = 0.
The overlap term can be rewritten in terms of the autocorrelation A_x(k) of x:
C(k) = sum_i h_i (1 - h_{i+k}) = (N - A_x(k)) / 4.
Minimizing max_k C(k) is equivalent to maximizing min_k A_x(k).
For any sequence x with sum 0, the sum of non-zero lag autocorrelations is
sum_{k != 0} A_x(k) = -A_x(0) = -sum(x_i^2) <= 0.
Therefore, min_{k != 0} A_x(k) <= 0.
The uniform solution h_i = 0.5 corresponds to x_i = 0, for which A_x(k) = 0 for all k.
This achieves min_k A_x(k) = 0, which is the global maximum possible.
Thus, the uniform solution is the unique optimal solution (up to measure zero).
"""

import numpy as np

N = 100

def construct():
    """
    Construct the optimal step heights.
    
    Returns:
        np.ndarray: An array of N step heights, all equal to 0.5.
    """
    # The uniform distribution h_i = 0.5 is the theoretical optimum.
    # It minimizes the maximum overlap to N/4.
    h = np.full(N, 0.5)
    
    # Verification of constraints (optional but good for robustness)
    # 1. Bounds
    assert np.all(h >= 0.0)
    assert np.all(h <= 1.0)
    # 2. Sum constraint
    # For N=100, sum is exactly 50.0
    assert abs(np.sum(h) - N / 2.0) < 1e-12
    
    return h
