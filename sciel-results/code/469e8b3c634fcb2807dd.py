"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item (larger = more attractive)
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure we have the right shape
    if prizes.ndim == 2 and prizes.shape[0] == 5 and prizes.shape[1] == 100:
        p = prizes
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        p = prizes.reshape(1, 100)
    else:
        flat = prizes.reshape(-1)
        if flat.shape[0] >= 100:
            p = flat[:100].reshape(1, 100)
        else:
            p = np.zeros((1, 100))
            p[0, :flat.shape[0]] = flat
    
    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        w = weights
    elif weights.ndim == 1 and weights.shape[0] == 100:
        w = weights.reshape(1, 100)
    else:
        flat = weights.reshape(-1)
        if flat.shape[0] >= 100:
            w = flat[:100].reshape(1, 100)
        else:
            w = np.zeros((1, 100))
            w[0, :flat.shape[0]] = flat
    
    # Compute aggregate metrics
    total_prize = p.sum(axis=0)  # (100,)
    total_weight = w.sum(axis=0)  # (100,)
    max_weight = w.max(axis=0)  # (100,) - bottleneck constraint weight
    
    # Core ratio: total prize / total weight
    ratio = total_prize / (total_weight + 1e-9)
    
    # Light bottleneck penalty: items heavy in any single constraint limit
    # how many other items can be added. Use a mild multiplicative penalty.
    # (1.0 / (1.0 + max_weight)) ranges from ~0.67 to 1.0 for weights in [0,1]
    bottleneck_factor = 1.0 / (1.0 + max_weight)
    
    # Combined score
    base_score = ratio * bottleneck_factor
    
    # Sharpen with power 2.0 - balances focus on top items while keeping
    # enough spread for ACO exploration
    scores = base_score ** 2.0
    
    # Ensure all scores are positive
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
