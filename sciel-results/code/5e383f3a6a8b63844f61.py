# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 280.0  # Safety margin before 300s/330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    Uses a BOHN-style approach: treats cost as an additional dimension by
    dynamically estimating lambda from budget remaining and best value.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Estimate lambda: dynamic trade-off between function value and cost
    # lambda scales with how "expensive" remaining budget is relative to best value
    # If f_best is large, we can afford more cost; if small, we should be cheaper
    f_best_abs = abs(f_best) if f_best != 0 else 1e-6
    
    # Cost statistics
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-9
    cost_norm = (cost - cost_mean) / cost_std
    
    # Dynamic lambda: higher when f_best is large (more budget to spend)
    # Lower when f_best is close to 0 (should be frugal)
    lambda_base = 0.3 * f_best_abs / (f_best_abs + 1.0)
    
    # Compute expected improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute expected cost-adjusted improvement
    # Model: effective objective = f(x) + lambda * cost(x)
    # We want to minimize this, so we maximize negative of it
    # EI on the combined objective
    
    # Approximate: treat cost as having some uncertainty
    # Use a heuristic: the "expected" cost contribution is lambda * cost
    # The "uncertainty" in cost is modeled as proportional to cost_std
    
    # Combined expected improvement
    # E[improvement] = EI_f - lambda * E[improvement in cost]
    # For simplicity, assume cost is deterministic per candidate
    
    # Score = EI - lambda * cost (normalized)
    cost_normalized = (cost - np.min(cost)) / (np.max(cost) - np.min(cost) + 1e-9)
    
    # Adaptive lambda: increase exploration when sigma is high
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    sigma_norm = (sigma - sigma_mean) / sigma_std
    
    # Lambda varies with uncertainty: explore more when uncertain
    lambda_adaptive = lambda_base * (1.0 + 0.5 * np.clip(sigma_norm, -1, 1))
    
    # Final score: EI minus cost penalty
    scores = ei - lambda_adaptive * cost_normalized
    
    # Add a small exploration bonus for high-uncertainty regions
    exploration_bonus = 0.05 * sigma_norm
    
    scores = scores + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
