"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: return pure exploitation if we're running out of time
    if time.time() - _start_time > 320:
        return -mu
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei[~np.isfinite(ei)] = 0.0
    
    # Structural change: blend EI with a UCB-style exploration term.
    # Early on, exploration matters more to escape local minima in the
    # multimodal Michalewicz landscape. Later, exploitation dominates.
    # This is different from pure UCB (current) or pure EI (inspiration 1).
    #
    # The blend weight shifts from exploration-heavy to exploitation-heavy
    # over the course of 25 acquisition calls (30 total evals - 5 initial).
    
    # UCB score for minimization: -mu + beta * sigma
    # We want to combine EI (which naturally balances) with an explicit
    # exploration bonus that decays.
    
    # Dynamic exploration weight: starts high, decays to near zero
    # At call 1: w=1.0 (full exploration bonus)
    # At call 25: w~0.1 (mostly exploitation)
    w = max(0.1, 1.0 - 0.04 * (_call_count - 1))
    
    # Scale EI to be comparable to the UCB term.
    # EI values are typically in [0, f_best - min_mu], while UCB term
    # is in [-max_mu + beta*max_sigma, ...]. 
    # A simple approach: normalize EI by the range of mu, then add
    # a scaled exploration term.
    
    # Actually, let's use a simpler structural idea:
    # score = EI + alpha * (sigma / (sigma + eps)) * (f_best - mu_max)
    # But this is getting complicated. Let's use a cleaner blend:
    
    # score = (1 - w) * normalized_EI + w * normalized_UCB
    # where normalized means scaled to [0, 1] roughly.
    
    # Simpler: just add a decaying exploration bonus to EI.
    # The bonus is proportional to sigma, scaled by the spread of mu.
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-12 else 1.0
    exploration_bonus = w * (sigma_safe / mu_range) * mu_range * 0.5
    # Simplifies to: w * sigma_safe * 0.5
    
    # Wait, that's just 0.5 * w * sigma. Let me think differently.
    
    # Better structural idea: use a "knowledge gradient" style approach
    # where we weight EI by the probability that the point is the best.
    # But that's expensive. 
    #
    # Alternative structural change: use a two-stage score.
    # Primary: EI. Secondary (tie-breaker/exploration): sigma.
    # score = EI + lambda * sigma, where lambda decays.
    
    lambda_explore = w * 0.5 * mu_range / (np.mean(sigma_safe) + 1e-12)
    # This normalizes the exploration term to be on the same scale as EI.
    
    score = ei + lambda_explore * sigma_safe
    
    # Ensure finite
    score[~np.isfinite(score)] = 0.0
    
    return score
# EVOLVE-BLOCK-END
