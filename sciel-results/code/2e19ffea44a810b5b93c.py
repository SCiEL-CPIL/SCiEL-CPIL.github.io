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
        if flat.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, flat.shape[0])
            tmp[:n] = flat[:n]
            flat = tmp
        p = flat.reshape(1, 100)
    
    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        w = weights
    elif weights.ndim == 1 and weights.shape[0] == 100:
        w = weights.reshape(1, 100)
    else:
        flat = weights.reshape(-1)
        if flat.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, flat.shape[0])
            tmp[:n] = flat[:n]
            flat = tmp
        w = flat.reshape(1, 100)
    
    # Total prize per item
    total_prizes = p.sum(axis=0)
    
    # Total weight per item
    total_weights = w.sum(axis=0)
    
    # Per-constraint efficiency: prize_c / weight_c for each constraint
    # Avoid division by zero
    eff = p / (w + 1e-9)
    
    # Use soft-min across constraints: items that are efficient in ALL constraints
    # are preferred. Soft-min with temperature T approximates min as T->0
    # Using log-sum-exp trick for numerical stability
    T = 0.5
    log_eff = np.log(eff + 1e-15)
    # soft-min = -T * log(mean(exp(-log_eff/T)))
    # = -T * (log(sum(exp(-log_eff/T))) - log(n_constraints))
    neg_log_eff_div_T = -log_eff / T
    max_neg = neg_log_eff_div_T.max(axis=0, keepdims=True)
    log_sum_exp = max_neg.squeeze(0) + np.log(np.exp(neg_log_eff_div_T - max_neg).sum(axis=0))
    n_constraints = eff.shape[0]
    soft_min = -T * (log_sum_exp - np.log(n_constraints))
    
    # Combine: total prize ratio with soft-min efficiency
    # Items with high total prize AND good per-constraint efficiency score high
    ratio = total_prizes / (total_weights + 1e-9)
    
    # Blend: use geometric mean of ratio and soft_min to balance
    scores = (ratio ** 1.5) * (soft_min ** 1.5)
    
    # Sharpen further
    scores = scores ** 1.2
    
    # Ensure all scores are positive (task says non-positive become 1e-9)
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
