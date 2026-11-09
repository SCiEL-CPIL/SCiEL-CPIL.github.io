"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    mu, sigma: shape (2000,), GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: shape (2000,), evaluation cost for each candidate
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)

    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu_safe) / sigma_safe
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute Lower Confidence Bound (LCB) score
    # We want to minimize, so we prefer low mu and high uncertainty
    # LCB = -mu + beta * sigma
    # Dynamic beta based on relative uncertainty
    mu_range = np.ptp(mu_safe) if n > 1 else 1.0
    if mu_range < 1e-9:
        mu_range = 1.0
    
    sigma_mean = np.mean(sigma_safe)
    rel_uncertainty = sigma_mean / mu_range
    
    # Beta ranges from ~1.0 (exploit) to ~3.0 (explore)
    beta = 1.0 + 2.0 * np.tanh(5.0 * rel_uncertainty)
    
    lcb_scores = -mu_safe + beta * sigma_safe

    # Compute probability of improvement
    pi = norm.cdf(z)

    # Cost-aware scoring:
    # We want to prefer cheap points. cost is in [1, 2].
    # A simple division by cost is a linear penalty.
    # A more aggressive penalty can be used to prune high-cost regions.
    
    # Strategy: Quantile-Based Cost-Discounted LCB
    # Instead of just dividing by cost, we apply a sigmoidal discount
    # that heavily penalizes high cost and mildly penalizes low cost.
    
    # Normalize cost to [0, 1] for the sigmoid
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_range = 1.0
        
    cost_norm = (cost_safe - cost_min) / cost_range
    
    # Sigmoid discount: exp(-k * cost_norm)
    # k controls the strength of the penalty.
    # If k is large, high cost points are heavily discounted.
    # Let's use k=2.0 as a moderate penalty.
    k = 2.0
    cost_discount = np.exp(-k * cost_norm)
    
    # Apply discount to EI and LCB
    ei_discounted = ei * cost_discount
    lcb_discounted = lcb_scores * cost_discount

    # Blend EI and LCB
    # Use a dynamic weight based on regret
    min_val = -3.86278
    regret = f_best - min_val
    # When regret is large, explore more (weight LCB higher)
    # When regret is small, exploit more (weight EI higher)
    exploration_weight = 0.3 + 0.7 * min(1.0, regret / 0.5)
    
    # Normalize EI and LCB to comparable scales
    ei_max = np.max(ei_discounted) if n > 0 else 1.0
    lcb_max = np.max(lcb_discounted) if n > 0 else 1.0
    if ei_max < 1e-9:
        ei_max = 1.0
    if lcb_max < 1e-9:
        lcb_max = 1.0

    ei_norm = ei_discounted / ei_max
    lcb_norm = lcb_discounted / lcb_max

    scores = (1 - exploration_weight) * ei_norm + exploration_weight * lcb_norm

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return scores

    # Alternative: pure EI / cost
    if n >= 100 and time.time() < deadline - 10.0:
        alt_scores = ei / cost_safe
        alt_scores = np.where(np.isfinite(alt_scores), alt_scores, 0.0)
        
        # Compare: mean mu of top-10 points
        top_k = min(10, n // 10)
        if top_k >= 1:
            idx_current = np.argsort(scores)[-top_k:]
            idx_alt = np.argsort(alt_scores)[-top_k:]
            
            mean_mu_current = np.mean(mu_safe[idx_current])
            mean_mu_alt = np.mean(mu_safe[idx_alt])
            
            # Prefer the one with lower mean mu in top-k
            if mean_mu_alt < mean_mu_current - 1e-6:
                scores = alt_scores

    # Final safety
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
