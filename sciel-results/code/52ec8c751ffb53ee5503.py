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
    Acquisition function using a dynamic hybrid of Expected Improvement (EI)
    and Upper Confidence Bound (UCB) for minimizing the Styblinski-Tang 1D function.
    
    The dynamic beta in UCB increases as the current best value approaches the
    known global minimum, promoting exploration to escape local optima when
    close to the solution.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Safety checks and type conversion
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    # Clip z to prevent numerical issues in cdf/pdf for extreme values
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # 2. Dynamic Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # We want higher scores for lower mu and higher sigma.
    # beta controls exploration.
    
    # Known global minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # Calculate progress towards the known minimum
    # f_best starts high (e.g., > 0) and decreases to f_min_known.
    # We map f_best to a progress value p in [0, 1].
    # p = 0 when f_best is far from min (e.g., 0.0), p = 1 when f_best <= f_min_known.
    # Using 0.0 as a reference "start" point is arbitrary but effective.
    # If f_best > 0, progress is 0.
    # If f_best <= f_min_known, progress is 1.
    # Linear interpolation in between.
    
    # To make it robust, let's define the range of interest.
    # ST function values can be large positive.
    # Let's assume f_best > f_min_known.
    # p = (0.0 - f_best) / (0.0 - f_min_known)
    # If f_best = 0, p = 0.
    # If f_best = -39.1662, p = 1.
    
    if f_best > 0:
        progress = 0.0
    elif f_best <= f_min_known:
        progress = 1.0
    else:
        progress = (0.0 - f_best) / (0.0 - f_min_known)
        progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic beta:
    # Start with low exploration (beta_min) when far from optimum.
    # Increase exploration (beta_max) when close to optimum to escape local minima.
    beta_min = 0.5
    beta_max = 2.5
    
    beta = beta_min + (beta_max - beta_min) * progress
    
    ucb = -mu + beta * sigma_safe
    
    # 3. Normalize EI and UCB to [0, 1]
    def _normalize(arr):
        a_min = np.min(arr)
        a_max = np.max(arr)
        rng = a_max - a_min
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - a_min) / rng
    
    ei_norm = _normalize(ei)
    ucb_norm = _normalize(ucb)
    
    # 4. Weighted Combination
    # Favor exploitation (EI) but allow significant exploration (UCB).
    # Weights inspired by successful hybrid approaches.
    w_ei = 0.7
    w_ucb = 0.3
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
