"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    i = np.arange(N)
    center = N / 2.0

    # Raised cosine window: 0 at edges, 1 at center
    # This should reduce the peak of the autoconvolution
    heights = 0.5 * (1.0 + np.cos(np.pi * (i - center) / (N / 2.0)))
    heights = np.maximum(heights, 0.0)

    # Normalize so that the sum is reasonable (optional, but keeps scale consistent)
    # The metric is likely scale-invariant, so this doesn't matter, but let's keep it simple.

    return list(heights)

# EVOLVE-BLOCK-END
