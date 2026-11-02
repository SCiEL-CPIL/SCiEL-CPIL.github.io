import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    # The objective is to minimize max(|autocorr|) / (sum(x))^2.
    # For x in {-1, 1}, autocorr(0) = sum(x^2) = N.
    # Thus max(|autocorr|) >= N.
    # The denominator is (sum(x))^2.
    # To minimize the ratio, we must maximize (sum(x))^2.
    # The maximum possible sum is N (all 1s) or -N (all -1s).
    # Max (sum(x))^2 = N^2.
    # Minimum possible ratio = N / N^2 = 1/N.
    # Any other sequence has |sum(x)| < N, so (sum(x))^2 < N^2,
    # resulting in a larger ratio (since numerator is still >= N).
    # Therefore, the constant sequence is the global optimum.
    return np.ones(N, dtype=int)
# EVOLVE-BLOCK-END
