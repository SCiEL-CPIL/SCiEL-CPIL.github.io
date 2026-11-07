"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 320.0  # safety margin before 330s kill


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Combines:
    1. Scarcity-weighted geometric mean (from inspiration 1): emphasizes
       efficiency in tighter constraints.
    2. Soft-min surrogate via log-sum-exp with temperature (smooth bottleneck):
       penalizes items that are inefficient in any constraint, but smoothly.
    3. Total prize scaling to maintain comparable magnitude.
    
    The soft-min uses a temperature parameter to control how sharply we
    penalize the worst constraint. Lower temperature = closer to hard min.
    """
    # Deadline guard
    if time.time() - _START > _DEADLINE - 5.0:
        # Fallback: simple prize-to-total-weight ratio
        P = np.sum(prizes, axis=0)
        W = np.sum(weights, axis=0)
        return np.maximum(P / (W + 1e-9), 1e-9)
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    eps = 1e-9
    capacity = 25.0
    n_constraints, n_items = weights.shape
    
    # Per-constraint efficiency: prize/weight for each item in each constraint
    eff = prizes / (weights + eps)  # (5, 100)
    
    # Scarcity-based constraint weighting:
    # Constraints with less total available weight relative to capacity are tighter
    weight_sum = weights.sum(axis=1)  # (5,)
    # Tightness: how close total weight is to capacity (higher = tighter)
    tightness = weight_sum / capacity  # (5,)
    # Scarcity weight: inverse of remaining capacity fraction
    remaining_frac = np.maximum(1.0 - tightness, eps)  # (5,)
    scarcity = 1.0 / remaining_frac  # (5,)
    scarcity = scarcity / scarcity.sum()  # normalize to sum to 1
    
    # Weighted log-efficiency for scarcity-weighted geometric mean
    log_eff = np.log(eff + eps)  # (5, 100)
    
    # Scarcity-weighted geometric mean
    weighted_log = (scarcity[:, None] * log_eff).sum(axis=0)  # (100,)
    scarcity_weighted_geo = np.exp(weighted_log)  # (100,)
    
    # Soft-min surrogate: use log-sum-exp with temperature to approximate min
    # log(min_c eff_c) ≈ -T * log-sum-exp(-eff_c / T) for small T
    # This smoothly penalizes items with any very low efficiency
    temperature = 0.1  # controls sharpness of the soft-min
    
    # Compute soft-min of efficiency across constraints
    # soft_min_c(eff_c) = -T * log(sum_c exp(-eff_c / T))
    neg_eff_div_T = -eff / temperature  # (5, 100)
    # log-sum-exp for numerical stability
    max_neg = np.max(neg_eff_div_T, axis=0)  # (100,)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_eff_div_T - max_neg[:, None]), axis=0))  # (100,)
    soft_min_eff = -temperature * log_sum_exp  # (100,)
    
    # Combine: blend scarcity-weighted geometric mean with soft-min
    # Normalize both to have mean 1 for fair blending
    sg_mean = np.mean(scarcity_weighted_geo)
    sm_mean = np.mean(soft_min_eff)
    
    sg_norm = scarcity_weighted_geo / (sg_mean + eps)
    sm_norm = soft_min_eff / (sm_mean + eps)
    
    # Blend: 50% scarcity-weighted geo, 50% soft-min
    combined = 0.5 * sg_norm + 0.5 * sm_norm
    
    # Scale by total prize to maintain comparable magnitude and favor high-value items
    total_prize = prizes.sum(axis=0)  # (100,)
    heuristic = combined * total_prize
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
