"""Optimized program for second autocorrelation inequality task."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a Sidon set to minimize the maximum autoconvolution peak
    # relative to the L2 norm, assuming L1 normalization.
    # Sidon set in [0, 49]: {0, 1, 3, 7, 15, 31, 48}
    # Size K = 7.
    # Max multiplicity M = 2.
    # Ratio = K / M = 3.5.
    sidon_indices = [0, 1, 3, 7, 15, 31, 48]
    K = len(sidon_indices)
    h = [0.0] * N
    for idx in sidon_indices:
        h[idx] = 1.0 / K
    return h
# EVOLVE-BLOCK-END
