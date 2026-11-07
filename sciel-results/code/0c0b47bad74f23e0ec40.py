"""Acquisition function for Bayesian optimization of Beale 2D using EI with adaptive kappa."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement with adaptive exploration.
    
    For the Beale function (min at (3, 0.5), value 0), we use:
    - Early: higher kappa (more exploration) to find the right basin
    - Late: lower kappa (more exploitation) to refine near the minimum
    
    The Beale function has a narrow valley, so we need to balance
    exploration (finding the valley) with exploitation (narrowing in).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa based on f_best
    # Beale function: starts around 10-20, minimum is 0
    # When f_best is large, explore more; when small, exploit more
    if f_best > 5.0:
        kappa = 1.5
    elif f_best > 1.0:
        kappa = 1.0
    elif f_best > 0.1:
        kappa = 0.5
    elif f_best > 0.01:
        kappa = 0.2
    else:
        kappa = 0.1
    
    # Expected Improvement with kappa
    # EI_kappa = (f_best + kappa*sigma - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best + kappa*sigma - mu) / sigma
    # This is equivalent to EI with an offset
    
    z = (f_best + kappa * sigma - mu) / sigma
    ei_kappa = (f_best + kappa * sigma - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei_kappa = np.where(np.isfinite(ei_kappa), ei_kappa, 0.0)
    
    # Add tiny noise to break ties
    ei_kappa += 1e-10 * np.random.randn(len(ei_kappa))
    
    return ei_kappa
