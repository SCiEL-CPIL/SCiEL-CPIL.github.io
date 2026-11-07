"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples a single objective value from the Gaussian posterior for each candidate,
    normalized by the evaluation cost, to balance exploration and exploitation.
    """
    deadline = time.time() + 295.0  # Safety margin before 300s limit
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: sample from posterior N(mu, sigma)
    # Use a single random draw scaled by sigma for each candidate
    rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    sampled_values = mu + sigma_safe * z
    
    # For minimization, lower sampled values are better
    # We want to maximize the score, so we invert: higher score = lower sampled value
    # Normalize by cost: score = -sampled_value / cost
    # But we want higher scores to be better, so:
    # score = (f_best - sampled_value) / cost  (expected improvement style)
    # Actually, for Thompson sampling in minimization:
    # We pick the candidate with the lowest sampled value
    # So our score should be higher when sampled value is lower
    # score = -sampled_value / cost_safe
    
    # To keep scores positive and comparable, shift by f_best:
    # score = (f_best - sampled_value) / cost_safe
    # This is like a stochastic version of expected improvement
    
    scores = (f_best - sampled_values) / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
