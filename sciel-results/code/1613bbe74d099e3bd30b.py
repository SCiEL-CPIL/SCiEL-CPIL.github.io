# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.5 * sigma_safe
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB for minimization
    # We want to maximize score, so higher score means more likely to be better
    # UCB: -mu + beta * sigma (for minimization, we want low mu, so -mu is high when mu is low)
    # But we need to balance exploration and exploitation
    
    # Adaptive beta: start with exploration, shift to exploitation as we get closer to optimum
    # Estimate progress based on how close mu is to f_best
    # If f_best is close to the theoretical minimum (0 for Weierstrass), we're doing well
    # Use the fraction of points where mu < f_best as a proxy for convergence
    
    # A simple adaptive strategy:
    # beta = 1.0 initially, decrease as we find better points
    # But we don't have history, so use the relationship between mu and f_best
    
    # Alternative: use a fixed but well-tuned beta
    # From the observations, UCB with beta=1.0 scored 0.817 (worse than EI's 1.0)
    # EI scored 1.0, and a combined approach scored 1.5
    
    # Let's try a hybrid: normalize both EI and UCB, then combine
    # The key insight: early on, explore (UCB); later, exploit (EI)
    
    # Use the spread of sigma to adapt beta
    # If sigma is large everywhere, we're still exploring -> higher beta
    # If sigma is small, we're exploiting -> lower beta
    
    mean_sigma = np.mean(sigma_safe)
    std_sigma = np.std(sigma_safe)
    
    # Adaptive beta: higher when uncertainty is high
    # Scale beta based on the relative uncertainty
    if mean_sigma > 0:
        # beta between 0.5 and 2.0
        # When mean_sigma is large relative to f_best, explore more
        # But f_best could be negative, so use absolute value or a reference
        beta = 1.0 + 0.5 * (std_sigma / (mean_sigma + 1e-9))
        beta = np.clip(beta, 0.5, 2.0)
    else:
        beta = 1.0
    
    # UCB score for minimization
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to [0, 1] for combining
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # Combine: weight EI more as we converge
    # Use the fraction of points with mu < f_best as convergence indicator
    frac_better = np.mean(mu < f_best)
    
    # As frac_better increases, we're finding better points, so exploit more
    # weight_ei increases from 0.5 to 0.9
    weight_ei = 0.5 + 0.4 * frac_better
    weight_ucb = 1.0 - weight_ei
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
