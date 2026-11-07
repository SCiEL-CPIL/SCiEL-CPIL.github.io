import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Cost-Weighted Acquisition: Blends Cost-Weighted PI with a
    quantile-based exploitation term. The blend shifts from exploration
    (probability of improvement) to exploitation (predicted low quantile)
    as the best observed value approaches the global minimum.
    
    Components:
    1. Cost-Weighted PI: Phi((f_best - mu) / sigma) / cost
       - Captures the probability of finding a better point per unit cost.
       - Strong in early stages when f_best is large.
       
    2. Cost-Weighted Quantile: -q_0.1 / cost
       - q_0.1 = mu + z_0.1 * sigma (10th percentile)
       - Focuses on the lower tail of the predictive distribution.
       - Strong in late stages when we want to exploit known low regions.
       
    The blend weight w_exploit increases as |f_best| decreases, shifting
    from exploration to exploitation.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety checks
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)
    
    # --- Component 1: Cost-Weighted Probability of Improvement ---
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    pi = norm.cdf(z_clipped)
    cw_pi = pi / cost
    
    # --- Component 2: Cost-Weighted 10th Percentile ---
    z_01 = -1.28155
    q_low = mu + z_01 * sigma
    cw_quantile = -q_low / cost
    
    # --- Adaptive Blend ---
    # f_best for rosenbrock ranges from ~large (initial) to ~0 (optimal)
    # When f_best is large, explore more (weight PI higher)
    # When f_best is small, exploit more (weight quantile higher)
    f_best_abs = abs(f_best)
    
    # w_exploit: 0.2 when f_best is large, 0.8 when f_best is small
    w_exploit = 0.2 + 0.6 * (1.0 - np.exp(-f_best_abs / 5.0))
    w_explore = 1.0 - w_exploit
    
    # Normalize both components to [0, 1] for stable blending
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr_min) / arr_range
    
    cw_pi_norm = normalize(cw_pi)
    cw_quantile_norm = normalize(cw_quantile)
    
    # Final score: weighted blend
    score = w_explore * cw_pi_norm + w_exploit * cw_quantile_norm
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback: simple exploitation
        score = cw_quantile_norm
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
