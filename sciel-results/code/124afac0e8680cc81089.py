"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best-fit: prefer bins that leave the least remaining capacity.
    # Only consider bins with enough remaining capacity (bins >= item)
    rem = bins - item
    # For valid bins (rem >= 0), we want to minimize rem (best fit)
    # So score = -rem for valid bins, and -inf for invalid bins
    scores = np.where(rem >= 0, -rem, -np.inf)
    return scores
# EVOLVE-BLOCK-END
