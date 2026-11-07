import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0

def _deadline_reached():
    return time.time() - _start_time > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Expected Hypar Improvement (EHI) with Gaussian damping for exploration control.
    
    mu: array of shape (2000,) - GP mean in objective units
    sigma: array of shape (2000,) - GP std dev in objective units
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each point, 1 + mean(x)
    
    Returns: array of shape (2000,) with acquisition scores (higher = better)
    """
    if _deadline_reached():
        # Return a simple EI-based fallback to stay within time
        mu_safe = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
        sigma_safe = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
        sigma_safe = np.maximum(sigma_safe, 1e-12)
        z = (f_best - mu_safe) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei
    
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Hypar Improvement (EHI) for minimization
    # EHI = E[hypar(f(x) - f_best)] / cost
    # hypar(t) = 1 - exp(-t^2 / (2*c^2)) for t > 0, 0 otherwise
    # This is a bounded, smooth version of improvement that saturates
    # The Gaussian damping factor exp(-sigma^2/(2*c^2)) suppresses over-exploration
    
    # Scale parameter for the hypar function
    # Use a fraction of the range of observed values for robustness
    # Since we're minimizing, improvement is f_best - f(x)
    # We want EHI to be meaningful relative to the problem scale
    
    # Estimate a reasonable scale from the data
    # Use the standard deviation of mu as a proxy for problem scale
    mu_scale = np.std(mu) + 1e-6
    # Also consider the gap between f_best and the mean
    gap = np.abs(f_best - np.mean(mu)) + 1e-6
    # Combine: use the larger of the two to avoid overly aggressive saturation
    c = max(mu_scale, gap * 0.5)
    
    # Compute the hypar improvement expectation
    # For a Gaussian N(mu, sigma^2), E[1 - exp(-(f_best - f)^2/(2*c^2))] * I(f < f_best)
    # This can be computed analytically or approximated
    
    # Analytical form:
    # Let Z = (f_best - f) / sigma, so f = f_best - sigma*Z
    # E[1 - exp(-sigma^2*Z^2/(2*c^2))] * I(Z > 0)
    # = Phi(-z0) - E[exp(-sigma^2*Z^2/(2*c^2)) * I(Z > 0)] where z0 = 0
    # Actually, let's use a different approach
    
    # The expected hypar improvement:
    # E[hypar] = E[(1 - exp(-(f_best - f)^2/(2*c^2))) * I(f < f_best)]
    
    # Using the identity: for X ~ N(mu, sigma^2),
    # E[exp(-a*(f_best - X)^2) * I(X < f_best)] can be computed
    
    # Let's use a numerical-free approximation:
    # EHI = (f_best - mu) * Phi(z) * damping + sigma * phi(z) * damping
    # where z = (f_best - mu) / sigma
    # and damping = exp(-sigma^2 / (2 * c^2))
    
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Standard EI-like term
    ei_term = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_term = np.maximum(ei_term, 0.0)
    
    # Gaussian damping: suppresses points with high uncertainty
    # This prevents over-exploration of regions where sigma is very large
    damping = np.exp(-sigma**2 / (2 * c**2))
    
    # EHI score before cost adjustment
    ehi = ei_term * damping
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # Use a power > 1 to more aggressively penalize expensive points
    cost_adjusted = ehi / (cost ** 1.5)
    
    # Ensure all scores are finite and non-negative
    cost_adjusted = np.nan_to_num(cost_adjusted, nan=0.0, posinf=1e6, neginf=0.0)
    cost_adjusted = np.maximum(cost_adjusted, 0.0)
    
    return cost_adjusted
