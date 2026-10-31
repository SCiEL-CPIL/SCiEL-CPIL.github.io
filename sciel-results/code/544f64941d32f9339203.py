"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = preferred bin.
    
    Strategy: Polarization (Smooth transition from Worst-Fit to Best-Fit).
    - For small items (< 50), prefer Worst Fit (maximize remaining capacity) to keep bins "open" for medium items.
    - For large items (> 50), prefer Best Fit (minimize remaining capacity) to save space.
    - Transition is smooth using a sigmoid function centered at 50.
    """
    capacity = 100.0
    threshold = 0.5 * capacity  # 50.0
    scale = 10.0  # Transition width
    
    # Compute remaining capacity
    remaining = bins - item
    
    # Mask for bins that can fit the item
    can_fit = remaining >= 0
    
    # Sigmoid weight: 0 for small items, 1 for large items
    x = (item - threshold) / scale
    # Clip x to avoid overflow in exp
    x_clipped = np.clip(x, -50, 50)
    w = 1.0 / (1.0 + np.exp(-x_clipped))
    
    # Score: (1 - 2*w) * remaining
    # If w=0 (small item): Score = remaining (Worst Fit: max remaining is best)
    # If w=1 (large item): Score = -remaining (Best Fit: min remaining is best)
    scores = (1.0 - 2.0 * w) * remaining
    
    # Set infeasible bins to -inf
    scores = np.where(can_fit, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
