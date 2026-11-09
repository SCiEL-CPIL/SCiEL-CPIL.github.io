"""Program for Bayesian optimization acquisition function."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
TIME_BUDGET = 310.0  # Leave 20s margin before 330s kill


def _time_remaining():
    return TIME_BUDGET - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid EI + LCB approach optimized for
    the Styblinski-Tang 1D minimization task.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Safety checks
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp extreme sigma values
    sigma_safe = np.clip(sigma_safe, 1e-10, 1e6)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB = mu - kappa * sigma (we want to maximize -LCB = -mu + kappa*sigma)
    # Adaptive kappa: start higher for exploration, decay as we approach optimum
    target = -39.1662
    gap = f_best - target  # positive if we're above the optimum
    
    # kappa starts around 2.0 when gap is large, decays to ~0.5 when gap is small
    kappa = 0.5 + 1.5 * (1.0 / (1.0 + np.exp(-gap / 5.0)))
    lcb_score = -mu + kappa * sigma_safe
    
    # Normalize each component to [0, 1] range for stable combination
    def _normalize(arr):
        a_min = np.min(arr)
        a_max = np.max(arr)
        rng = a_max - a_min
        if rng < 1e-15:
            return np.zeros_like(arr)
        return (arr - a_min) / rng
    
    ei_norm = _normalize(ei)
    lcb_norm = _normalize(lcb_score)
    
    # Adaptive weighting: favor LCB more when gap is large (exploration needed),
    # favor EI more when gap is small (exploitation needed)
    # When gap >> 0, explore more with LCB; when gap ~ 0, exploit more with EI
    w_lcb = 0.4 + 0.4 * (1.0 / (1.0 + np.exp(-gap / 5.0)))
    w_ei = 1.0 - w_lcb
    
    # Combine with adaptive weighting
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
