"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: Array of shape (2000,) with posterior means.
        sigma: Array of shape (2000,) with posterior standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (2000,) with evaluation costs.
        
    Returns:
        Array of shape (2000,) with scores (higher is better).
    """
    t0 = time.time()
    deadline = t0 + 290.0  # Leave safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive to avoid division by zero or negative std
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: Sample a function value from the posterior at each candidate
    # y_i ~ N(mu_i, sigma_i)
    # We want to minimize y_i / cost_i, so we maximize -y_i / cost_i
    # Since we can only return scores, and the loop picks the argmax, 
    # returning -sampled_value / cost works.
    
    # However, pure Thompson Sampling can be too exploratory early on or too greedy late on.
    # A common variant is to sample y and then apply a small cost-aware adjustment.
    # Given the task emphasizes "cost-aware" and the previous attempt failed with complex heuristics,
    # let's stick to a clean TS formulation but ensure it's robust.
    
    # Generate random normal samples
    # To be efficient, we generate the noise once and scale it.
    noise = np.random.randn(n)
    samples = mu + sigma * noise
    
    # The score should reflect the expected improvement or utility.
    # In TS, we simply pick the point with the lowest sampled value (for minimization).
    # To incorporate cost, we divide by cost.
    # score_i = -samples[i] / cost[i]
    
    # One potential issue: If f_best is very low, samples might be positive,
    # leading to negative scores. This is fine as long as they are finite and relative.
    # However, if we want to be more strictly "improvement" based, we could use:
    # score_i = -(samples[i] - f_best) / cost[i]
    # This centers the scores around 0 relative to current best.
    # Let's use the centered version to keep magnitudes manageable and intuitive.
    
    scores = -(samples - f_best) / cost
    
    # Check for non-finite values
    if not np.all(np.isfinite(scores)):
        # Fallback to a simple UCB-like score if something went wrong
        scores = (-mu + 2.0 * sigma) / cost
        # Ensure finite
        scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
