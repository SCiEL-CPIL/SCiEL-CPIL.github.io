import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIME_LIMIT = 295.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Being the Minimum (PBM).
    
    This is a structural change from EI/UCB hybrids. PBM calculates the
    probability that each candidate is the global minimum among the set of
    candidates, assuming independent Gaussian posteriors. This encourages
    selecting points that are likely to be the best in the current batch,
    balancing exploitation (low mu) and exploration (high sigma reduces
    the confidence in others being better, but also increases uncertainty
    in self).
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        return -(mu + 1.0 * sigma_safe)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    n = len(mu)
    
    # Compute pairwise Z-scores for P(f_i < f_j)
    # Z_ij = (mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2)
    
    mu_col = mu[:, np.newaxis]
    mu_row = mu[np.newaxis, :]
    
    D = mu_row - mu_col
    
    sigma_sq_col = sigma_safe[:, np.newaxis]**2
    sigma_sq_row = sigma_safe[np.newaxis, :]**2
    S = np.sqrt(sigma_sq_row + sigma_sq_col)
    
    Z = D / S
    Z = np.clip(Z, -10, 10)
    
    # Log probability that f_i < f_j
    # P(f_i < f_j) = Phi(Z_ij)
    # We want sum over j != i of log(Phi(Z_ij))
    
    log_probs = norm.logcdf(Z)
    
    # Zero out diagonal (i=j)
    idx = np.arange(n)
    log_probs[idx, idx] = 0.0
    
    scores = np.sum(log_probs, axis=1)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
