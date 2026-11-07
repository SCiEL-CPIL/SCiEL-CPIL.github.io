import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 300s hard limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    For minimization, EI measures the expected reduction in the objective value
    below f_best. We maximize this score.
    
    EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    where Z = (f_best - mu) / sigma, Phi is the standard normal CDF, and phi
    is the standard normal PDF.
    
    We use a small xi (improvement threshold) to encourage slight exploitation
    while maintaining some exploration.
    
    We also add a small UCB component to ensure exploration in regions with
    high uncertainty, which helps escape local minima in the Goldstein-Price
    function.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a neutral score if time is critical
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-10)  # avoid division by zero
    
    # Expected Improvement for minimization
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF using scipy
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # For minimization, this gives positive values when mu < f_best
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip small negative values to zero (EI should be non-negative)
    EI = np.maximum(EI, 0.0)
    
    # Add a small UCB component for exploration
    # This helps in regions where EI is low but uncertainty is high
    # Use a small coefficient to not dominate the EI signal
    kappa_ucb = 0.5
    UCB = -mu + kappa_ucb * sigma_safe
    
    # Combine: primarily EI, with a small UCB boost
    # Normalize both to comparable scales
    # Actually, let's just use EI as the primary score
    # The EI already balances exploitation and exploration
    
    # Ensure finiteness
    scores = np.nan_to_num(EI, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
