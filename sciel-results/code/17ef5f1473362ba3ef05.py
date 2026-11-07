"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO on a 5-constraint knapsack.
    
    Strategy: 'constraint regret' metric using soft-min (log-sum-exp based)
    of per-constraint efficiency ratios, combined with prize magnitude.
    
    For each constraint c, item i's efficiency is prize_i / weight_ic.
    We compute a soft-min across constraints of these efficiencies to penalize
    items that are poor in any single constraint, while keeping the heuristic
    smooth for better ACO probability gradients.
    
    The final score is: prize_i * soft_min_c(prize_i / (weight_ic + eps))^alpha
    where alpha > 1 amplifies the penalty for being weak in any constraint.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected shape (5, 100)")
    
    if not (np.all(np.isfinite(prizes)) and np.all(np.isfinite(weights))):
        raise ValueError("Non-finite values in input")
    
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    
    # Per-constraint efficiency: prize_i / weight_ic
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    efficiency = prizes / (weights + eps)  # shape (5, 100)
    
    # Soft-min across constraints using log-sum-exp trick
    # soft_min(x) = -log(sum(exp(-x)))
    # This is a smooth approximation to min(x)
    neg_eff = -efficiency  # shape (5, 100)
    # For numerical stability, subtract max before exp
    max_neg = np.max(neg_eff, axis=0, keepdims=True)  # shape (1, 100)
    shifted = neg_eff - max_neg  # shape (5, 100), all <= 0
    exp_shifted = np.exp(shifted)  # shape (5, 100)
    log_sum = np.log(np.sum(exp_shifted, axis=0))  # shape (100,)
    soft_min = -log_sum + np.squeeze(max_neg)  # shape (100,)
    
    # Clip to positive values
    soft_min = np.maximum(soft_min, eps)
    
    # Prize magnitude: use geometric mean of prizes across constraints
    # (each item has a prize in each constraint dimension)
    prize_geo_mean = np.exp(np.mean(np.log(prizes + eps), axis=0))  # shape (100,)
    
    # Final heuristic: prize magnitude * soft_min efficiency^alpha
    alpha = 1.5
    heuristic = prize_geo_mean * (soft_min ** alpha)
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
