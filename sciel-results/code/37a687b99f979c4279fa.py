"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None
_DEADLINE = 310.0  # 330s limit, leave 20s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value
    cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the limit, return a quick fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: simple probability-weighted improvement per cost
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        prob = norm.cdf(z)
        prob = np.clip(prob, 0.0, 1.0)
        fallback = prob * np.maximum(f_best - mu, 0.0) / np.maximum(cost, 1e-8)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Probability that mu < f_best (improvement is possible)
    prob_improve = norm.cdf(z_clipped)
    prob_improve = np.clip(prob_improve, 0.0, 1.0)
    
    # Expected improvement (for minimization)
    ei = (f_best - mu) * prob_improve + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Safe cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Primary: Probability-weighted improvement per unit cost
    # This is P(mu < f_best) * (f_best - mu) / cost
    # It directly maximizes expected regret reduction per unit cost
    prob_weighted = prob_improve * np.maximum(f_best - mu, 0.0) / cost_safe
    
    # Secondary: EI per unit cost (standard cost-aware EI)
    ei_cost = ei / cost_safe
    
    # Normalize both components using z-scoring for comparable scales
    def zscore(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-9:
            return arr - mean
        return (arr - mean) / std
    
    pw_norm = zscore(prob_weighted)
    ei_norm = zscore(ei_cost)
    
    # Combine: weight the probability-weighted term more heavily as it's
    # more directly aligned with the knapsack-style cost-aware objective
    # Use 60% probability-weighted, 40% EI/cost
    w_pw = 0.6
    w_ei = 0.4
    
    scores = w_pw * pw_norm + w_ei * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
