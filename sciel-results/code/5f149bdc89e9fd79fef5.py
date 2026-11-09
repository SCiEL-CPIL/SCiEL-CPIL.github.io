"""Cost-aware acquisition for Bayesian optimization on three-hump camel 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # safety margin before 330s kill

def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Linear Cost Penalty and Additive Exploration.
    
    Structure: Score = EI - lambda * cost + gamma * sigma
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions in objective units
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best (lowest) observed value
        cost: array of shape (2000,), evaluation costs
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        # Fallback: Simple EI
        sigma_safe = np.maximum(sigma, 1e-10)
        delta = f_best - mu
        z = np.clip(delta / sigma_safe, -10, 10)
        ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    f_best = float(f_best)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z to avoid numerical issues with very large/small values
    z_clipped = np.clip(z, -10, 10)
    
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Linear Cost Penalty + Additive Exploration
    # Based on knowledge: "linear subtraction allows high-EI expensive points to compete"
    # and "sigma/scale exploration bonus aids in the small-budget regime"
    
    # Coefficients tuned for the 3H Camel problem scale (values ~0-10, cost ~1-2)
    LAMBDA_COST = 0.5   # Penalty per unit of cost
    GAMMA_EXPL = 1.0    # Exploration weight
    
    score = ei - LAMBDA_COST * cost_safe + GAMMA_EXPL * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
