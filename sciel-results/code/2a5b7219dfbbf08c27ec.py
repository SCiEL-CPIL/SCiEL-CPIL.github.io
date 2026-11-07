import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: blend of Cost-Weighted PI and a UCB-like exploration term,
    with adaptive exploitation weight based on the current best value.
    
    The key structural change: instead of a single acquisition criterion, we use a
    mixture that adapts its exploitation/exploration balance based on how good f_best is.
    Early on (f_best is large), we explore more; later (f_best is small), we exploit more.
    
    Components:
    1. Cost-Weighted PI: Phi((f_best - mu) / sigma) / cost
    2. Cost-Weighted UCB: (f_best - mu + beta * sigma) / cost, where beta is adaptive
    3. A pure exploitation term: -mu / cost, weighted by how close f_best is to 0
    
    The blend weight shifts from exploration to exploitation as f_best decreases.
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
    
    # Compute z for PI
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    pi = norm.cdf(z_clipped)
    
    # Cost-weighted PI
    cw_pi = pi / cost
    
    # Adaptive beta for UCB: larger when f_best is large (early), smaller when f_best is small
    # f_best for rosenbrock ranges from ~large (initial) to ~0 (optimal)
    # Scale: when f_best > 10, beta ~ 2; when f_best ~ 0, beta ~ 0.5
    f_best_abs = abs(f_best)
    beta = 0.5 + 1.5 * np.exp(-f_best_abs / 10.0)
    
    # Cost-weighted UCB (in terms of improvement: we want low mu, high sigma)
    # UCB for minimization: f_best - mu + beta * sigma (higher is better improvement potential)
    ucb_improvement = (f_best - mu + beta * sigma)
    # Normalize by cost
    cw_ucb = ucb_improvement / cost
    # Clip to avoid extreme values
    cw_ucb = np.clip(cw_ucb, -100, 100)
    
    # Exploitation term: prefer low mu and low cost
    exploitation = -mu / cost
    exploitation = np.clip(exploitation, -100, 100)
    
    # Adaptive blend: weight of exploitation increases as f_best approaches 0
    # When f_best is large, w_exploit ~ 0.3; when f_best ~ 0, w_exploit ~ 0.7
    w_exploit = 0.3 + 0.4 * (1.0 - np.exp(-f_best_abs / 5.0))
    w_explore = 1.0 - w_exploit
    
    # Normalize each component to [0, 1] range for stable blending
    # For cw_pi: already in [0, 1]
    # For cw_ucb and exploitation: normalize by their respective ranges
    
    # Simple normalization: subtract min, divide by range
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr_min) / arr_range
    
    cw_pi_norm = cw_pi  # already [0,1]
    cw_ucb_norm = normalize(cw_ucb)
    exploitation_norm = normalize(exploitation)
    
    # Final score: weighted blend
    score = w_exploit * exploitation_norm + w_explore * (0.6 * cw_pi_norm + 0.4 * cw_ucb_norm)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback: simple exploitation
        score = exploitation_norm
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
