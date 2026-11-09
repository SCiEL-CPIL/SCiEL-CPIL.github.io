import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 325.0
_EPS = 1e-12

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a diversity-aware
    exploration bonus to prevent collapse into local minima of the RBF mixture.
    
    The score is:
        score = EI + alpha * sigma_normalized
    
    where EI is the standard Expected Improvement for minimization, and
    sigma_normalized encourages exploration of uncertain regions.
    
    Args:
        mu: (2000,) posterior mean
        sigma: (2000,) posterior std
        f_best: float, lowest observed value
    
    Returns:
        (2000,) scores. Higher is better.
    """
    # Deadline check
    if time.time() - _T0 > _DEADLINE:
        return -(mu - f_best)
    
    n = len(mu)
    
    # Ensure inputs are finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma), sigma, _EPS)
    sigma_safe = np.maximum(sigma_safe, _EPS)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi(d) + sigma(x) * phi(d)
    # where d = (f_best - mu(x)) / sigma(x)
    d = (f_best - mu_safe) / sigma_safe
    
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    ei = (f_best - mu_safe) * Phi_d + sigma_safe * phi_d
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < _EPS:
        ei_norm = np.zeros(n)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    # Normalize sigma to [0, 1]
    sig_min = np.min(sigma_safe)
    sig_max = np.max(sigma_safe)
    sig_range = sig_max - sig_min
    if sig_range < _EPS:
        sig_norm = np.zeros(n)
    else:
        sig_norm = (sigma_safe - sig_min) / sig_range
    
    # Adaptive alpha: start with more exploration, shift to exploitation
    # Use a simple heuristic: if f_best is close to the minimum, exploit more
    # We don't know the true minimum, but we can use the range of mu as a proxy
    mu_range = np.max(mu_safe) - np.min(mu_safe)
    if mu_range < _EPS:
        alpha = 0.3
    else:
        # If f_best is near the lower end of mu range, we're likely close to optimum
        # Use a sigmoid-like transition
        progress = (f_best - np.min(mu_safe)) / mu_range
        progress = np.clip(progress, 0, 1)
        # Early: more exploration (alpha ~ 0.5), Late: more exploitation (alpha ~ 0.1)
        alpha = 0.1 + 0.4 * (1 - progress)
    
    # Combined score
    score = ei_norm + alpha * sig_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
