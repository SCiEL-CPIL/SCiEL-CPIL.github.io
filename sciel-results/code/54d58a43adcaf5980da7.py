"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-Fit: prefer bins that leave the most remaining capacity.
    # This helps preserve tight fits for future items and reduces fragmentation.
    remaining = bins - item
    # Assign -inf to bins that cannot hold the item
    scores = np.where(remaining >= 0, remaining, -np.inf)
    return scores
# EVOLVE-BLOCK-END
