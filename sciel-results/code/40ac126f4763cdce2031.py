"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to ensure we finish well within the 300s limit
_START_TIME = time.time()
DEADLINE_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining_budget():
    return DEADLINE_LIMIT - _elapsed() - SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """
    Standard Expected Improvement for minimization.
    EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu - xi) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _probability_of_improvement(mu, sigma, f_best):
    """
    Probability of Improvement for minimization: P(f(x) < f_best).
    PI = Phi((f_best - mu) / sigma)
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using Thompson sampling.
    
    Uses Thompson sampling: draw a random sample from the posterior for each candidate,
    then select the candidate that maximizes the sampled value divided by cost.
    This introduces probabilistic exploration that escapes local exploitation.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Check deadline - if we're running out of time, return a simple score
    if _remaining_budget() < 1.0:
        # Fallback: simple cost-adjusted probability of improvement
        pi = _probability_of_improvement(mu, sigma, f_best)
        score = pi / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Thompson sampling: sample from the posterior
    # For each candidate, sample y ~ N(mu, sigma^2)
    # We want to find candidates with low expected values (good for minimization)
    # So we score by: -sampled_value / cost (higher is better)
    
    # Draw a single random normal for each candidate
    z = np.random.randn(len(mu))
    sampled_values = mu + sigma * z
    
    # Score: we want low sampled values (good for minimization), divided by cost
    # To make higher = better, we negate
    # But we also want to balance exploration vs exploitation
    # The Thompson sample naturally balances this:
    # - Where sigma is high (uncertain), the sample varies more -> exploration
    # - Where mu is low (good), the sample tends to be low -> exploitation
    
    # Score = -sampled_value / cost
    # This means candidates with low sampled values AND low cost get high scores
    score = -sampled_values / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
