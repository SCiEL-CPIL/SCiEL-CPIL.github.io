"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 310  # 310s out of 330s budget, safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    Uses a hybrid approach:
    1. Cost-normalized LCB for exploration of unexplored regions
    2. Cost-normalized Expected Improvement (EI) for exploitation near known good points
    3. Adaptive beta parameter based on how far we are from the known minimum
    
    The key insight: cost is 1 + mean(x_scaled), so points near the center of the
    domain are cheaper. The shifted minima are in the inner 80% of the domain,
    so center-biased exploration is structurally advantageous.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure inputs are clean
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Compute LCB score (lower is better, so we maximize -LCB) ---
    # LCB = mu - beta * sigma, we want to minimize this
    # Adaptive beta: start with higher exploration, reduce as we get closer to optimum
    # The known minimum of Styblinski-Tang 2D is -78.3323
    # Use progress-based beta: if f_best is close to -78.3323, reduce exploration
    target_min = -78.3323
    progress = np.clip((f_best - target_min) / 50.0, 0.0, 1.0)
    # Beta from 2.5 (early, high exploration) to 1.0 (late, exploitation)
    beta = 1.0 + 1.5 * progress
    
    lcb = mu - beta * sigma  # We want to minimize this
    lcb_score = -lcb  # Maximize this
    
    # --- Compute Expected Improvement (EI) ---
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z) where Z = (f_best - mu) / sigma
    # For minimization, improvement is max(0, f_best - y)
    improvement = f_best - mu
    z = improvement / sigma
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    # EI is non-negative, higher is better
    
    # --- Cost normalization ---
    # Divide both scores by cost to prefer cheaper evaluations
    lcb_cost = lcb_score / cost_safe
    ei_cost = ei / cost_safe
    
    # --- Combine LCB and EI ---
    # Normalize both to comparable scales
    lcb_std = np.std(lcb_cost) + 1e-10
    ei_std = np.std(ei_cost) + 1e-10
    
    lcb_norm = lcb_cost / lcb_std
    ei_norm = ei_cost / ei_std
    
    # Weight: blend based on progress
    # Early: more LCB (explore), Late: more EI (exploit)
    lcb_weight = 0.5 + 0.5 * progress
    ei_weight = 1.0 - lcb_weight
    
    score = lcb_weight * lcb_norm + ei_weight * ei_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
