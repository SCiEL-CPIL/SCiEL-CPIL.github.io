# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

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
    
    # 3. Compute PES-like score (Probability of Improvement * Info Gain)
    prob_improve = norm.cdf(z)
    info_gain = np.log1p(sigma_safe)
    pes_score = prob_improve * (1.0 + 1.5 * info_gain)
    
    # 4. Normalize all scores to [0, 1]
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    pes_norm = normalize(pes_score)
    
    # 5. Adaptive weighting based on convergence
    # frac_better: fraction of points where mu < f_best
    # As this increases, we're finding better points, so exploit more
    frac_better = np.mean(mu < f_best)
    
    # Transition from exploration (UCB/PES) to exploitation (EI)
    # Use call count for a smoother schedule
    x = _call_count - 1
    mid = 12
    k = 0.4
    w_ei = 1.0 / (1.0 + np.exp(-k * (x - mid)))
    
    # Blend with convergence indicator
    w_ei = 0.5 * w_ei + 0.5 * (0.3 + 0.7 * frac_better)
    w_ei = np.clip(w_ei, 0.1, 0.9)
    
    # Remaining weight split between UCB and PES
    w_remaining = 1.0 - w_ei
    # Early on, favor UCB for exploration; later, favor PES for global minimum discovery
    w_ucb = w_remaining * (1.0 - 0.5 * w_ei)
    w_pes = w_remaining * (0.5 * w_ei)
    
    score = w_ei * ei_norm + w_ucb * ucb_norm + w_pes * pes_norm
    
    # 6. Add a small exploration bonus for points with high sigma
    # This helps escape local minima in the acquisition landscape
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / n
    score += 0.05 * sigma_rank
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
