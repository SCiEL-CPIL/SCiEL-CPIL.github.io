import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0 # seconds

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    This function computes the EI score for each candidate point, given the
    posterior mean (mu) and standard deviation (sigma) from a Gaussian Process,
    and the current best observed value (f_best).
    
    EI is preferred over UCB here because the sphere function is unimodal and
    well-behaved. EI focuses on points that are likely to improve upon f_best,
    leading to faster convergence to the global minimum (0) in a limited
    evaluation budget (30 evaluations).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    # A small epsilon is used to handle points where the GP is certain (sigma=0)
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Check deadline. If we are close to the limit, fallback to pure exploitation
    # to ensure the loop finishes without hanging or crashing due to time limits.
    # Although EI on 2000 points is fast, this is a safety net.
    current_time = time.time()
    if current_time - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: maximize -mu (find the minimum predicted mean)
        score = -mu
        score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
        return score

    # Calculate Z-score: (f_best - mu) / sigma
    # Z represents how many standard deviations the current best is above the predicted mean.
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # For numerical stability:
    # If Z is very large, Phi(Z) ~ 1, phi(Z) ~ 0, so EI ~ f_best - mu.
    # If Z is very negative, Phi(Z) ~ 0, phi(Z) ~ 0, so EI ~ 0.
    
    # We can compute directly, but we must be careful with large Z.
    # scipy.stats.norm.cdf and pdf are stable for typical ranges.
    # Let's clip Z to avoid extreme values that might cause precision issues, 
    # though scipy is generally robust.
    
    # Calculate PDF and CDF
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # Calculate EI
    score = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Handle numerical artifacts
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure scores are finite
    return score
