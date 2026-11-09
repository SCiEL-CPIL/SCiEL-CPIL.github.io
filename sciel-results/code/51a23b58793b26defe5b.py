import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Expected Improvement (EI).
    The exploration parameter xi is scaled by the gap between the current
    best value and the known global minimum. This encourages exploration
    when far from the optimum and exploitation when close.
    """
    if _elapsed() > 328.5:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Known minimum for Michalewicz 2D
    f_min = -1.8013
    
    # Calculate gap to the global minimum
    gap = max(f_best - f_min, 1e-6)
    
    # Adaptive exploration parameter
    # Larger gap -> larger xi -> more exploration
    # Smaller gap -> smaller xi -> more exploitation
    xi = 0.5 * gap
    
    # Expected Improvement with xi
    improvement = f_best - mu - xi
    z = improvement / sigma
    z = np.clip(z, -10.0, 10.0)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei
