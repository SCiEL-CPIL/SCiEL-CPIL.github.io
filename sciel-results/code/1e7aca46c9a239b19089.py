"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a Worst-Fit style heuristic.
    The item goes to the highest-scoring bin.
    
    For worst-fit, we want to prefer bins with the most remaining capacity after
    placing the item, i.e., we score by the remaining capacity (bins - item).
    Higher score = more remaining capacity = preferred for worst-fit.
    
    Bins that cannot fit the item get a very low score (or -inf) so they are
    never selected unless no bin can fit the item.
    """
    remaining = bins - item
    # Bins that can't fit the item get a very negative score
    score = np.where(remaining >= 0, remaining, -np.inf)
    return score
# EVOLVE-BLOCK-END
