"""Hybrid acquisition: Thompson Sampling with adaptive exploration for Michalewicz 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling acquisition function with adaptive exploration.
    
    Combines the stochastic exploration of Thompson Sampling with an adaptive
    exploitation/exploration balance that responds to the current state of
    the optimization.
    
    Strategy:
    1. Compute a Thompson sample: s = mu + sigma * z, z ~ N(0,1)
    2. Blend this with a deterministic UCB-style score: ucb = -mu + beta * sigma
    3. The blend weight adapts based on how close mu is to f_best (exploitation
       when close) and the overall uncertainty level.
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior std in objective units
        f_best: float, lowest observed value so far
    
    Returns:
        scores: array of shape (2000,), finite, higher means evaluate next
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running low on time, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > 320:  # 10s safety margin before 330s kill
        return -mu + sigma
    
    # Ensure sigma is strictly positive and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Thompson sample
    z = np.random.standard_normal(len(mu))
    thompson_sample = mu + sigma * z
    
    # Compute adaptive UCB score
    # beta adapts based on overall uncertainty level
    mean_sigma = np.mean(sigma)
    
    # Michalewicz 2D has values roughly in [-1.8, 1.6], so scale ~ 3.4
    func_scale = 3.4
    uncertainty_ratio = mean_sigma / func_scale
    
    # Higher beta when uncertainty is high (early), lower when confident (late)
    beta = 1.5 + 1.5 * np.exp(-2.0 * uncertainty_ratio)
    
    ucb_score = -mu + beta * sigma
    
    # Compute expected improvement for additional signal
    z_ei = (f_best - mu) / sigma
    phi_z = norm.pdf(z_ei)
    Phi_z = norm.cdf(z_ei)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Normalize EI to [0, 1] for blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Compute exploitation pressure: how close is mu to f_best?
    # When mu is close to f_best, we should exploit more (trust the mean)
    # When mu is far, we should explore more (trust the uncertainty)
    mu_range = np.max(mu) - np.min(mu)
    if mu_range > 1e-12:
        # Normalize mu to [0, 1], where 0 = best (lowest mu)
        mu_norm = (mu - np.min(mu)) / mu_range
    else:
        mu_norm = np.zeros_like(mu)
    
    # Exploitation weight: higher when mu is close to f_best
    # Use the EI as a proxy for how promising each point is
    exploit_weight = 0.5 + 0.5 * ei_norm
    
    # Blend Thompson sample and UCB score
    # Thompson sample score (higher = lower sampled value)
    thompson_score = -thompson_sample
    
    # Normalize both scores to have zero mean and unit variance for blending
    def normalize_scores(s):
        s_mean = np.mean(s)
        s_std = np.std(s)
        if s_std > 1e-12:
            return (s - s_mean) / s_std
        return np.zeros_like(s)
    
    thompson_norm = normalize_scores(thompson_score)
    ucb_norm = normalize_scores(ucb_score)
    
    # Adaptive blend: more Thompson when uncertainty is high, more UCB when exploiting
    # Use the mean EI as a global signal for exploitation vs exploration
    mean_ei_norm = np.mean(ei_norm)
    
    # When mean EI is high, we're in exploration mode -> more Thompson
    # When mean EI is low, we're in exploitation mode -> more UCB
    thompson_weight = 0.4 + 0.4 * mean_ei_norm  # Range [0.4, 0.8]
    ucb_weight = 1.0 - thompson_weight
    
    # Final score is a weighted blend
    score = thompson_weight * thompson_norm + ucb_weight * ucb_norm
    
    # Add a small EI bonus to encourage finding better points
    ei_bonus = 0.1 * ei_norm
    score = score + ei_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
