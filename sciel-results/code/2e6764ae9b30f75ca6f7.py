# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill

_call_count = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _call_count, _last_f_best
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe
    
    # Reset logic for new seed/run
    if _last_f_best is not None and f_best > _last_f_best:
        _call_count = 0
    _last_f_best = f_best
    _call_count += 1
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute UCB for minimization
    # Adaptive beta based on uncertainty distribution
    mean_sigma = np.mean(sigma_safe)
    std_sigma = np.std(sigma_safe)
    
    if mean_sigma > 0:
        beta = 1.0 + 0.8 * (std_sigma / (mean_sigma + 1e-9))
        beta = np.clip(beta, 0.5, 2.5)
    else:
        beta = 1.0
    
    ucb = -mu + beta * sigma_safe
    
    # 3. Compute a novel structural component: 
    #    "Distance-weighted exploitation" - favor points that are both
    #    likely to improve AND are far from already-explored regions.
    #    We approximate exploration by looking at where sigma is high relative
    #    to the local neighborhood of mu values.
    
    # Sort by mu to identify local density of predictions
    sort_idx = np.argsort(mu)
    sorted_mu = mu[sort_idx]
    sorted_sigma = sigma_safe[sort_idx]
    
    # For each point, compute the average sigma of its 5 nearest neighbors
    # (in mu-space) to estimate local uncertainty concentration
    # This is O(n log n) with a sliding window approximation
    window = 5
    local_sigma_avg = np.zeros(n)
    
    # Use a simple approach: for each sorted position, average sigma of nearby points
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        local_sigma_avg[i] = np.mean(sorted_sigma[start:end])
    
    # Map back to original order
    local_sigma_avg_orig = np.empty(n)
    local_sigma_avg_orig[sort_idx] = local_sigma_avg
    
    # Points where sigma is much higher than local average are in "frontier" regions
    # This helps explore unexplored areas rather than re-exploring known uncertain regions
    frontier_score = np.where(local_sigma_avg_orig > 1e-9, 
                              sigma_safe / (local_sigma_avg_orig + 1e-9), 
                              1.0)
    frontier_score = np.clip(frontier_score, 0.0, 5.0)
    
    # 4. Normalize all scores to [0, 1]
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    frontier_norm = normalize(frontier_score)
    
    # 5. Adaptive weighting based on convergence
    # As we find better points (frac_better increases), shift to exploitation
    frac_better = np.mean(mu < f_best)
    
    # Use call count for a smoother schedule
    x = _call_count - 1
    mid = 10
    k = 0.5
    w_ei = 1.0 / (1.0 + np.exp(-k * (x - mid)))
    
    # Blend with convergence indicator
    w_ei = 0.4 * w_ei + 0.6 * (0.2 + 0.8 * frac_better)
    w_ei = np.clip(w_ei, 0.05, 0.95)
    
    # Remaining weight split between UCB and frontier exploration
    w_remaining = 1.0 - w_ei
    # Early on, favor frontier exploration; later, favor UCB
    w_ucb = w_remaining * (0.3 + 0.7 * w_ei)
    w_frontier = w_remaining * (1.0 - w_ucb / w_remaining) if w_remaining > 1e-9 else 0.0
    w_frontier = w_remaining - w_ucb
    
    score = w_ei * ei_norm + w_ucb * ucb_norm + w_frontier * frontier_norm
    
    # 6. Add a small exploration bonus for points with high sigma
    # This helps escape local minima in the acquisition landscape
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / n
    score += 0.03 * sigma_rank
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
