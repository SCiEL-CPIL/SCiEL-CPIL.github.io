"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses a structural change: Thompson Sampling with an exploitation prior.
This is a different optimization/search method from the EI/UCB hybrids in the inspirations.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 325.0  # 5s margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses Thompson Sampling: sample from the posterior at each candidate.
    Add an exploitation prior that boosts points where the posterior mean
    is already below f_best (i.e., promising regions).
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return EI as a safe default
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw one sample from N(mu, sigma) for each candidate
    # This gives a stochastic score; the candidate with the lowest sampled value
    # is the one most likely to be the best (for minimization)
    # We want higher score = evaluate next, so we negate the sample
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # For minimization, lower sample is better, so score = -sample
    ts_score = -samples
    
    # Exploitation prior: boost points where mu is already below f_best
    # This helps refine the neighborhood of the best-known point
    # The boost is proportional to how much better mu is than f_best,
    # but damped by sigma (high uncertainty reduces the boost)
    improvement = np.maximum(f_best - mu, 0.0)
    # Normalize improvement by sigma to get a z-like score
    z_improve = improvement / sigma_safe
    # Use a smooth, bounded boost: tanh(z_improve) * scale
    # This gives ~scale for very promising points, ~0 for non-promising
    exploit_boost = np.tanh(z_improve) * 0.5
    
    # Combine: Thompson Sampling score + exploitation boost
    # The TS score is in objective units (could be large), so we need to balance
    # Scale the TS score to be comparable to the boost
    # Use a running scale: the range of ts_score
    ts_range = np.ptp(ts_score) + 1e-9
    ts_normalized = (ts_score - np.min(ts_score)) / ts_range
    
    # Final score: normalized TS + exploitation boost
    scores = ts_normalized + exploit_boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
