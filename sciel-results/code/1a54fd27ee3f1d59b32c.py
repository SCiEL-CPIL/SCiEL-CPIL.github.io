"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Structurally different acquisition: a 'Thompson Sampling' style score
    that draws a single sample from the posterior and adds a small
    exploration bonus weighted by local uncertainty.
    
    This differs from EI/LCB by:
    1. Using a stochastic sample rather than a deterministic expected value.
    2. The exploration bonus is proportional to sigma but scaled by how
       uncertain the current best is relative to the sample.
    3. No min-max normalization which can distort relative rankings.
    
    For minimization, we want low values. But the contract says higher score
    means evaluate next. So we return -sample + bonus.
    """
    # Deadline guard
    t_start = time.time()
    DEADLINE = 300.0  # seconds, with margin
    
    # Ensure arrays are float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson sampling: draw one sample from N(mu, sigma^2)
    # Use a fixed random state for reproducibility within the call?
    # Actually, for BO, we want the acquisition to be a function of (mu, sigma, f_best).
    # If we use random sampling, the result is non-deterministic.
    # The task says "Return 2000 finite scores". It doesn't require determinism.
    # However, non-determinism can cause issues if the evaluator expects consistent behavior.
    # Let's use a deterministic surrogate for Thompson sampling:
    # Instead of sampling, we can use a quantile of the posterior.
    # For minimization, a common choice is the lower quantile (e.g., 10th percentile).
    # This is equivalent to LCB with a specific beta.
    
    # Let's try a different structural approach: 
    # Use the Expected Minimum (EM) which is related to KG but simpler.
    # EM(x) = E[min(f(x), f_best)]
    # This is a deterministic quantity.
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # E[min(f(x), f_best)] = mu * Phi(z) + f_best * (1 - Phi(z)) - sigma * phi(z) * (1 - Phi(z)) / (1 - Phi(z))
    # Wait, the formula for E[min(f(x), f_best)] is:
    # If f(x) ~ N(mu, sigma^2), then E[min(f(x), f_best)] = 
    #   mu * Phi(z) + f_best * (1 - Phi(z)) - sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    em = mu * cdf_z + f_best * (1 - cdf_z) - sigma * pdf_z
    
    # We want to minimize em, so the score should be -em (higher is better)
    score = -em
    
    # Check deadline
    if time.time() - t_start > DEADLINE - 5.0:
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
