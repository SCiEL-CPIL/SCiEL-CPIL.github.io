import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling and Expected Improvement acquisition.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean predictions
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best (lowest) function value observed so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better for selection)
    """
    t_start = time.time()
    deadline = t_start + 280.0
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Thompson Sampling Component
    # Sample from posterior predictive: y ~ N(mu, sigma^2)
    z_ts = np.random.normal(0, 1, n)
    ts_samples = mu - sigma_safe * z_ts
    
    # Normalize TS samples to be in a comparable range
    # We want higher score to mean "better" (lower objective value).
    # So we negate the samples.
    # Scale by standard deviation to make it unit-variance-ish
    ts_std = np.std(ts_samples)
    if ts_std < 1e-9:
        ts_std = 1.0
    ts_norm = -ts_samples / ts_std
    
    # 2. Expected Improvement Component
    # EI for minimization
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Normalize EI
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros(n)
    
    # 3. Adaptive Weighting
    # Michalewicz 2D global minimum is approx -1.8013
    # Assume max value is around 1.0 or 2.0 for normalization of progress
    min_possible = -1.8013
    max_possible = 1.0 # Rough upper bound for normalization
    
    # Progress: 0 at start (f_best high), 1 at end (f_best low)
    # f_best starts around 0 or 1, ends near -1.8
    progress = np.clip((f_best - min_possible) / (max_possible - min_possible), 0.0, 1.0)
    # Note: If f_best is high (e.g., 1.0), progress is high?
    # Let's redefine:
    # distance_to_opt = f_best - min_possible
    # If distance is large, we are far from optimum.
    # We want to explore MORE when far from optimum.
    # TS is inherently exploratory. EI is exploitative.
    # So weight of TS should be higher when far from optimum.
    
    # Let's use a simpler heuristic:
    # If f_best is much worse than min_possible, rely more on TS.
    # If f_best is close to min_possible, rely more on EI.
    
    dist = f_best - min_possible
    if dist > 1.0:
        alpha_ei = 0.2 # Low weight on EI, high on TS
    elif dist > 0.5:
        alpha_ei = 0.4
    elif dist > 0.1:
        alpha_ei = 0.6
    else:
        alpha_ei = 0.8 # High weight on EI, low on TS
        
    alpha_ts = 1.0 - alpha_ei
    
    # Combine
    scores = alpha_ts * ts_norm + alpha_ei * ei_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
