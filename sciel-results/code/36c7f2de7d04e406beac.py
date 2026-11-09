import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard state
_START = None
_DEADLINE_S = 300.0  # safety margin well under the 330s kill

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _deadline_ok():
    return _elapsed() < _DEADLINE_S

def acquisition(mu, sigma, f_best):
    global _START
    if _START is None:
        _START = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e9, neginf=-1e9)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    sigma = np.maximum(sigma, 1e-12)

    # Deadline guard: if we are close to the limit, return a safe fallback
    if not _deadline_ok():
        # Fallback: pure exploitation (lowest mu)
        return np.nan_to_num(-mu, nan=0.0, posinf=1e6, neginf=-1e6)

    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Handle cases where z is extremely large or small to avoid numerical issues
    # norm.cdf(z) -> 1 for large z, 0 for small z
    # norm.pdf(z) -> 0 for large |z|
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Add a small exploration term to encourage visiting uncertain regions
    # This helps if EI is zero due to numerical underflow or if the model is overconfident
    exploration_bonus = 0.1 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
