"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Hybrid heuristic: blend Best-Fit and Worst-Fit based on item size.
    # For small items, prefer bins with more remaining space (Worst-Fit-like)
    # to preserve tight fits for larger items. For large items, use Best-Fit.
    remaining = bins - item
    # Only consider bins with enough capacity (remaining >= 0)
    mask = remaining >= 0
    
    # Score: negative remaining (Best-Fit) blended with positive remaining (Worst-Fit)
    # Use a smooth blend: for small items, weight toward Worst-Fit; for large, Best-Fit.
    # Threshold at 25% of capacity (25 units).
    alpha = np.where(item < 25, 0.3, 0.7)  # 0.3 = more Worst-Fit, 0.7 = more Best-Fit
    
    # Best-Fit score: -remaining (lower remaining = higher score)
    # Worst-Fit score: +remaining (more remaining = higher score)
    # Blend: alpha * (-remaining) + (1 - alpha) * remaining
    # = -alpha * remaining + (1 - alpha) * remaining
    # = (1 - 2*alpha) * remaining
    
    blend_factor = 1 - 2 * alpha  # positive for Worst-Fit bias, negative for Best-Fit bias
    scores = blend_factor * remaining
    
    # Set infeasible bins to very low score
    scores = np.where(mask, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
