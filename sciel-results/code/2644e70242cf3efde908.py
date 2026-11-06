"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Hybrid best-fit with a small tie-breaking bonus for earlier bins.
    # The dominant term is the negative of the remaining capacity (best-fit),
    # which aggressively consolidates items to minimize the number of open bins.
    # A tiny linear index penalty breaks ties in favor of earlier bins,
    # which can reduce fragmentation over long sequences by keeping the
    # packing more "compact" in index space.
    remaining = bins - item
    # Small epsilon to avoid any numerical issues; also ensures we don't
    # accidentally prefer bins that are exactly full.
    remaining = np.maximum(remaining, 0.0)
    # Dominant best-fit signal
    score = -remaining
    # Tiny tie-breaker: prefer earlier bins by a negligible amount
    n = len(bins)
    if n > 1:
        # Scale the index penalty to be ~1e-9 of the typical remaining capacity
        # to ensure it only matters on near-ties
        index_penalty = np.arange(n, dtype=np.float64) * (1.0 / (n + 1)) * 1e-9
        score = score - index_penalty
    return score
# EVOLVE-BLOCK-END
