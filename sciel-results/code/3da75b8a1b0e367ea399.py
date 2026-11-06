"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Modified best-fit: prefer bins where remaining capacity after placement
    # is close to a target (40% of bin capacity), balancing tight fit with
    # avoiding fragmentation from overly tight packs.
    remaining = bins - item
    # Target: leave ~40 units of space (bin capacity is 100)
    target = 40.0
    # Quadratic penalty for deviation from target remaining
    scores = -((remaining - target) ** 2)
    # Only consider bins with enough capacity
    scores[remaining < 0] = -np.inf
    return scores
# EVOLVE-BLOCK-END
