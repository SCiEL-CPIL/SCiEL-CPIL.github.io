"""Rewritten program for the suite task; new heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = d.shape[0]
    pair = d[:, None] + d[None, :]
    rem = capacity - pair
    # Gaussian bump centred at a sum near capacity: rewards tight packing.
    sigma = 25.0
    val = np.exp(-((pair - capacity) ** 2) / (2.0 * sigma * sigma))
    # Only pairs that fit (with a small margin) are attractive.
    mask = rem >= 1
    out = np.where(mask, val, 1e-9)
    # Keep diagonal (i==i) non-positive so an item is never placed right after itself.
    out = np.where(np.eye(n, dtype=bool), 1e-9, out)
    return out
# EVOLVE-BLOCK-END
