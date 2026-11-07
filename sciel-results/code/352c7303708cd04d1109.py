"""Structural change: Probability of Improvement (PI) with adaptive noise floor."""
import numpy as np
from scipy.stats import norm

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) with an adaptive
    noise floor.
    
    PI measures the probability that a candidate is better than the current best.
    To prevent over-exploitation of low-variance regions and encourage exploration
    when the model is uncertain, we add an adaptive noise floor to the standard
    deviation. The floor decreases as f_best improves, allowing the search to
    become more precise near the optimum.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The best (lowest) observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher scores indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive noise floor:
    # Start with a larger floor to encourage exploration (by effectively increasing
    # uncertainty for points with small sigma, making PI less decisive).
    # As f_best decreases, reduce the floor to allow exploitation.
    # f_best for Ackley starts around 10-20 and goes to 0.
    # We want the floor to be significant early on.
    # Let's use a floor that is a fraction of f_best, with a minimum.
    noise_floor = np.maximum(0.1 * f_best, 1e-4)
    
    # Effective sigma
    sigma_eff = sigma_safe + noise_floor
    
    # Probability of Improvement: P(f(x) < f_best)
    # z = (f_best - mu) / sigma_eff
    z = (f_best - mu) / sigma_eff
    
    # PI = Phi(z)
    scores = norm.cdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
