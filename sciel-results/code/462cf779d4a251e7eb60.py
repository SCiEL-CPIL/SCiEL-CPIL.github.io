"""Optimized program for second autocorrelation inequality task."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # Uniform distribution normalized to L1 norm = 1.
    # For non-negative sequences, ||h*h||_inf = ||h||_2^2, so the ratio is 1/||h||_1.
    # With ||h||_1 = 1, the ratio is 1.
    h = [1.0 / N] * N
    return h
# EVOLVE-BLOCK-END
