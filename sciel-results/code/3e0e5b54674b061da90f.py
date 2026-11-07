"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next) for cost-aware acquisition.
    
    Uses a hybrid approach:
    1. Thompson Sampling-inspired stochastic sampling with cost normalization
    2. Expected Improvement per unit cost as a deterministic baseline
    3. Time-aware safety: if computation takes too long, fall back to simpler metric
    """
    start_time = time.time()
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Deadline guard: if we've already spent too much time, return simple EI/cost
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Simple fallback: EI per unit cost
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Strategy 1: Expected Improvement per unit cost (deterministic baseline)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_per_cost = ei / cost
    
    # Strategy 2: Thompson Sampling - sample from GP predictive
    # Use a single random draw for efficiency
    rng = np.random.default_rng()
    eps = rng.standard_normal(n)
    ts_samples = mu + sigma * eps
    
    # For minimization, we want low values, so we minimize ts_samples / cost
    # But acquisition returns higher = better, so we negate
    ts_score = -ts_samples / cost
    
    # Strategy 3: UCB-style with cost awareness
    # Balance exploration and exploitation with a beta parameter
    beta = 2.0  # Exploration weight
    ucb = mu + beta * sigma  # For minimization, lower is better
    ucb_per_cost = -ucb / cost
    
    # Strategy 4: Probability of improvement per cost
    # P(mu < f_best) = Phi((f_best - mu) / sigma)
    p_improve = norm.cdf((f_best - mu) / sigma)
    p_per_cost = p_improve / cost
    
    # Combine strategies with weighted average
    # Normalize each to [0, 1] range for combination
    def normalize(arr):
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei_per_cost)
    ts_norm = normalize(ts_score)
    ucb_norm = normalize(ucb_per_cost)
    p_norm = normalize(p_per_cost)
    
    # Weighted combination: favor EI/cost (proven) with some exploration
    w_ei = 0.40
    w_ts = 0.25
    w_ucb = 0.20
    w_p = 0.15
    
    combined = w_ei * ei_norm + w_ts * ts_norm + w_ucb * ucb_norm + w_p * p_norm
    
    # Ensure finite output
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Return the simpler, faster metric if running out of time
        scores = ei_per_cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    return combined
# EVOLVE-BLOCK-END
