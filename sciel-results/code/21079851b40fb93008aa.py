"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Combines Expected Improvement with adaptive LCB exploration,
    normalized and cost-discounted for robustness.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive LCB exploration
    # Estimate relative uncertainty to scale exploration
    med_sigma = np.median(sigma_safe)
    std_mu = np.std(mu) + 1e-8
    # beta scales with relative uncertainty: high sigma relative to mu spread -> more exploration
    # Also decay exploration as f_best improves (closer to minimum)
    f_scale = np.clip(f_best / 10.0, 0.0, 1.0)
    beta = np.clip(med_sigma / std_mu, 0.1, 2.0) * (0.5 + 0.5 * f_scale)
    
    # LCB for minimization: mu - beta * sigma (lower is better)
    lcb = mu - beta * sigma_safe
    # Negate for maximization
    lcb_score = -lcb
    
    # Normalize both components to be comparable
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score) + 1e-8
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # Weighted combination: 70% EI, 30% LCB
    # This balances exploitation (EI) with exploration (LCB)
    score = 0.7 * ei_norm + 0.3 * lcb_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Use a mild power to amplify cost preference without over-penalizing
    # Also add a dynamic penalty that increases as f_best improves (progress)
    # Early on: tolerate higher cost for high EI; later: penalize cost heavily
    cost_power = 0.5 + 0.5 * (1.0 - f_scale)  # Higher power when f_best is large (early)
    cost_factor = np.power(cost_safe, cost_power)
    score = score / cost_factor
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final safety: ensure no NaN or Inf
    if not np.all(np.isfinite(score)):
        score = np.zeros_like(score)
    
    return score
# EVOLVE-BLOCK-END
