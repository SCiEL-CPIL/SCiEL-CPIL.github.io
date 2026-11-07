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
    1. Scarcity-weighted marginal utility (from inspiration 1, scored 1.0422)
    2. Smooth bottleneck via log-sum-exp soft-min (from current program's family)
    3. Pareto dominance count as a structural bonus (per directive strategy)
    
    The core idea: identify items that are efficient in scarce constraints
    AND not dominated by other items in the prize/weight space.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle shape: prizes is (5, 100), weights is (5, 100)
    n_constraints, n_items = weights.shape
    
    eps = 1e-9
    capacity = 25.0
    
    # --- Component 1: Scarcity-weighted marginal utility (inspiration 1) ---
    # Total prize per item: (100,)
    P = prizes.sum(axis=0)
    
    # Per-constraint efficiency: (5, 100)
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = P[None, :] / (weights + eps)
    eff = np.where(np.isfinite(eff), eff, 0.0)
    
    # Top-k efficiency per constraint to estimate scarcity
    k = max(1, n_items // 5)  # top 20%
    top_k_eff = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_k_eff[c] = np.partition(eff[c], -k)[-k].mean()
    
    # Scarcity weights: constraints with higher top-k efficiency are scarcer
    scarcity = top_k_eff / (top_k_eff.sum() + eps)
    
    # Marginal utility: weighted sum of per-constraint efficiencies
    marginal_util = (scarcity[:, None] * eff).sum(axis=0)  # (100,)
    
    # --- Component 2: Smooth bottleneck via soft-min (current program family) ---
    # Geometric mean of per-constraint prize/weight ratios, but using
    # log-sum-exp for a smooth approximation of the minimum (bottleneck)
    # We use a soft-min: -log-sum-exp(-x) approximates min(x)
    # Here x = eff[c, i] for each constraint c and item i
    
    # Soft-min of efficiencies across constraints for each item
    # soft_min_i = -log(sum_c exp(-beta * eff[c, i])) / beta
    # As beta -> inf, this approaches min_c eff[c, i]
    beta = 5.0  # moderate smoothing
    neg_beta_eff = -beta * eff  # (5, 100)
    # For numerical stability, subtract max before exp
    max_neg_beta_eff = neg_beta_eff.max(axis=0, keepdims=True)
    exp_terms = np.exp(neg_beta_eff - max_neg_beta_eff)
    log_sum_exp = np.log(exp_terms.sum(axis=0)) + max_neg_beta_eff.squeeze(axis=0)
    soft_min_eff = -log_sum_exp / beta  # (100,)
    
    # Also compute the geometric mean as in the current program
    log_ratios = np.log(eff + eps)
    geo_mean = np.exp(log_ratios.mean(axis=0))  # (100,)
    
    # Combine soft-min and geo-mean for the bottleneck component
    # Normalize both to mean 1
    soft_min_norm = soft_min_eff / (soft_min_eff.mean() + eps)
    geo_norm = geo_mean / (geo_mean.mean() + eps)
    
    bottleneck_component = 0.5 * soft_min_norm + 0.5 * geo_norm
    
    # --- Component 3: Pareto dominance count (directive strategy) ---
    # For each item, count how many other items it dominates in the
    # (total_prize, total_weight) space.
    # Item i dominates item j if P[i] >= P[j] and W_total[i] <= W_total[j]
    # (with at least one strict inequality)
    
    W_total = weights.sum(axis=0)  # (100,)
    
    # Vectorized dominance count
    # For each item i, count j where P[i] >= P[j] and W_total[i] <= W_total[j]
    # and not (P[i] == P[j] and W_total[i] == W_total[j])
    
    # P[i] >= P[j]: (P[:, None] >= P[None, :]) -> (100, 100)
    prize_dominates = P[:, None] >= P[None, :]  # (100, 100)
    # W_total[i] <= W_total[j]: (W_total[:, None] <= W_total[None, :]) -> (100, 100)
    weight_dominates = W_total[:, None] <= W_total[None, :]  # (100, 100)
    
    # Dominance: both conditions true, and not identical
    identical = (P[:, None] == P[None, :]) & (W_total[:, None] == W_total[None, :])
    dominance = prize_dominates & weight_dominates & ~identical  # (100, 100)
    
    # Count for each item i: number of j it dominates
    dominance_count = dominance.sum(axis=1)  # (100,)
    
    # Normalize dominance count to have mean ~1
    # Items with high dominance count are on or near the Pareto front
    dominance_norm = dominance_count / (dominance_count.mean() + eps)
    
    # --- Combine all components ---
    # Base: marginal utility * base efficiency (from inspiration 1)
    base_eff = P / (W_total + eps)
    base_component = marginal_util * base_eff
    
    # Final heuristic: blend of base component, bottleneck, and dominance
    # Weight the components to balance efficiency, bottleneck, and structure
    # Use normalized versions for fair combination
    
    base_norm = base_component / (base_component.mean() + eps)
    
    # Final score: product of components to reward items that excel in all aspects
    # Or weighted sum? Product amplifies, sum is more robust.
    # Let's use a weighted geometric mean to balance
    # H = (base_norm^a * bottleneck_component^b * dominance_norm^c)^(1/(a+b+c))
    
    a, b, c = 0.5, 0.3, 0.2  # weights
    
    # Log-space combination for stability
    log_H = a * np.log(base_norm + eps) + b * np.log(bottleneck_component + eps) + c * np.log(dominance_norm + eps)
    H = np.exp(log_H)
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return H
    
    return H
# EVOLVE-BLOCK-END
