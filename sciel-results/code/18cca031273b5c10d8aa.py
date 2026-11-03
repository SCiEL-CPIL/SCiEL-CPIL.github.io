"""
Rewritten program for the Erdos minimum overlap task.

The task requires choosing step heights h in [0, 1] with sum N/2 to minimize
the maximum overlap of h with shifted 1-h.
The overlap at shift k is defined as S_k = sum_{i=0}^{N-1} h_i * (1 - h_{(i+k) % N}).

Mathematical Analysis:
1. The sum of overlaps over all shifts is sum_{k=0}^{N-1} S_k = sum_{i=0}^{N-1} h_i * (N - sum_{j=0}^{N-1} h_j).
   Given sum h_j = N/2, this equals sum h_i * (N/2) = (N/2) * (N/2) = N^2 / 4.
2. The average overlap is (N^2 / 4) / N = N / 4.
3. Therefore, the maximum overlap max_k S_k must be at least N / 4.
4. The constant sequence h_i = 0.5 yields S_k = sum 0.5 * 0.5 = N * 0.25 = N / 4 for all k.
5. Thus, the constant sequence achieves the theoretical lower bound and is the optimal solution.

This program constructs this optimal sequence.
"""
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def construct():
    """
    Returns the optimal sequence of step heights.
    
    Returns:
        np.ndarray: An array of N floats, all equal to 0.5.
    """
    # The optimal solution is the constant sequence h_i = 0.5.
    # This minimizes the maximum overlap to the theoretical minimum of N/4.
    return np.full(N, 0.5)

# EVOLVE-BLOCK-END
