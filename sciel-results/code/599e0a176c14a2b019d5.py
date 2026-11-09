"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Compute Expected Improvement for minimization."""
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _upper_confidence_bound(mu, sigma, kappa=1.0):
    """Compute UCB for minimization (lower is better)."""
    return mu - kappa * sigma

def _probability_of_improvement(mu, sigma, f_best, xi=0.0):
    """Compute Probability of Improvement for minimization."""
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma_safe
    return norm.cdf(z)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Returns 2000 scores, higher meaning evaluate next.
    """
    global _start_time
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple, fast acquisition to avoid timeout
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Determine the step number from the number of unique values in mu
    # This is a heuristic to adapt the acquisition function over time
    # Early on: more exploration. Later: more exploitation.
    
    # Compute basic statistics
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Estimate how far we are from the optimum
    # For sphere 1d on [-5, 5], f(x) = x^2, so f >= 0
    # If f_best is close to 0, we should be very exploitative
    progress = 1.0 - min(f_best / 25.0, 1.0)  # 25 = max value of x^2 on [-5,5]
    
    # Adaptive parameters
    # Early: high exploration. Late: high exploitation.
    # Use a smooth transition
    
    # For EI: xi should decrease as we get closer to the optimum
    xi = 0.05 * (1.0 - progress)
    
    # For UCB: kappa should decrease as we get closer to the optimum
    kappa = 2.0 * (1.0 - progress) + 0.5
    
    # Compute multiple acquisition functions
    ei = _expected_improvement(mu, sigma, f_best, xi=xi)
    ucb = _upper_confidence_bound(mu, sigma, kappa=kappa)
    poi = _probability_of_improvement(mu, sigma, f_best, xi=xi)
    
    # Normalize each to [0, 1]
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    poi_norm = normalize(poi)
    
    # Weight the acquisition functions
    # Early: weight UCB and POI more for exploration
    # Late: weight EI more for exploitation
    w_ei = 0.5 + 0.3 * progress
    w_ucb = 0.3 * (1.0 - progress) + 0.1
    w_poi = 0.2 * (1.0 - progress) + 0.1
    
    # Normalize weights
    w_sum = w_ei + w_ucb + w_poi
    w_ei /= w_sum
    w_ucb /= w_sum
    w_poi /= w_sum
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_poi * poi_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
