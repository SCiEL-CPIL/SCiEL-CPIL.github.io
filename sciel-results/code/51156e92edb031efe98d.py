"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Thompson Sampling and Expected Improvement.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: track start time to ensure we finish well within limits
    t_start = time.time()
    
    n = len(mu)
    
    # Sanitize inputs to ensure finite outputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # --- Component 1: Expected Improvement (EI) per unit cost ---
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_per_cost = ei / cost
    
    # --- Component 2: Thompson Sampling Score per unit cost ---
    # Draw a sample from the posterior: y = mu + sigma * epsilon
    # We want to minimize y, so we score by -y / cost
    # To reduce variance and make it more stable, we can use a single random draw
    # or a deterministic approximation. Let's use a single draw for true TS.
    # However, pure TS can be noisy. A common hybrid is to use the mean of the sample
    # or blend it. But the directive asks for Thompson Sampling.
    # Let's draw one sample for all points.
    rng = np.random.default_rng()
    epsilon = rng.standard_normal(n)
    y_sample = mu + sigma * epsilon
    
    # Score is -y_sample / cost (higher is better, i.e., lower y is better)
    ts_per_cost = -y_sample / cost
    
    # --- Component 3: Uncertainty Penalty / Exploration Bonus ---
    # Sometimes we want to explicitly reward high uncertainty if EI is low.
    # But EI already captures this. Let's stick to a blend of EI and TS.
    
    # Hybrid: Combine EI and TS scores.
    # EI is deterministic and stable. TS is stochastic.
    # A simple average or weighted sum works.
    # Let's use a weight of 0.5 for each.
    score = 0.5 * ei_per_cost + 0.5 * ts_per_cost
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time
    t_end = time.time()
    # The computation is O(n) with vectorized ops, so it should be very fast (<1ms).
    # No need for early termination unless n is huge, but n=2000 is small.
    
    return score
# EVOLVE-BLOCK-END
