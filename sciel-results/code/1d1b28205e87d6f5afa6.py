"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the ant colony heuristic.
    
    Strategy: Use a weighted harmonic mean of prize/weight ratios across all constraints.
    This penalizes items that are inefficient in any single tight constraint, acting as
    a smooth surrogate for the min-ratio objective. It respects the global resource
    distribution better than simple sum or max-based heuristics, guiding ants toward
    items that are efficient across ALL constraints.
    """
    # Handle input shapes: prizes and weights should be (5, 100)
    if prizes.ndim == 2:
        p = prizes  # (5, 100)
    else:
        # If prizes is 1D (100,), broadcast to (5, 100)
        p = np.tile(prizes, (weights.shape[0], 1))
    
    w = weights  # (5, 100)
    
    # Compute ratio for each constraint: prize / weight
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    ratios = p / (w + eps)  # (5, 100)
    
    # Weighted harmonic mean across constraints
    # Harmonic mean: n / sum(1/ratio_i)
    # This is equivalent to: 1 / mean(1/ratio_i)
    # We use a soft version to handle potential numerical issues and provide smoothness
    
    # Inverse ratios
    inv_ratios = 1.0 / (ratios + eps)  # (5, 100)
    
    # Mean of inverse ratios
    mean_inv = np.mean(inv_ratios, axis=0)  # (100,)
    
    # Harmonic mean = 1 / mean_inverse
    h = 1.0 / (mean_inv + eps)  # (100,)
    
    # Ensure positive values as per contract: "Heuristic entries that are not positive become 1e-9"
    h = np.where(h > 0, h, 1e-9)
    
    # Optionally normalize to keep values in a reasonable range
    # This helps with numerical stability in the ant colony algorithm
    max_h = np.max(h)
    if max_h > 0:
        h = h / max_h
    
    # Ensure minimum value
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
