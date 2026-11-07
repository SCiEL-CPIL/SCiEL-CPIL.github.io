"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track progress across calls within a seed
_last_f_best = np.inf
_step_count = 0

def _compute_ei(mu, sigma, f_best, xi=0.0):
    """Standard Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _compute_ucb(mu, sigma, f_best, kappa):
    """Lower Confidence Bound (UCB) - maximized for minimization."""
    return mu - kappa * sigma

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses an adaptive strategy that transitions from exploration to exploitation.
    """
    global _last_f_best, _step_count
    
    # Detect new seed or reset step count
    # If f_best drops significantly, it's likely a new seed or a major improvement
    # However, f_best is non-increasing. If it's much larger than previous, it's a new seed.
    # The problem states 3 seeds.
    if f_best > _last_f_best + 1.0:
        _step_count = 0
    else:
        _step_count += 1
        
    _last_f_best = f_best
    
    # Cap step count
    if _step_count > 30:
        _step_count = 30
        
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate EI
    ei = _compute_ei(mu, sigma, f_best, xi=0.0)
    
    # Calculate UCB with adaptive kappa
    # Early on: high kappa for exploration
    # Late on: low kappa for exploitation
    # Progress from 0 to 1 over ~25 steps (30 total evals - 5 initial)
    progress = min(_step_count / 25.0, 1.0)
    
    # Kappa schedule:
    # Start with high exploration (kappa ~ 5.0)
    # End with low exploitation (kappa ~ 1.0)
    # Linear interpolation
    kappa = 5.0 - 4.0 * progress
    
    ucb = _compute_ucb(mu, sigma, f_best, kappa)
    
    # Normalize both to [0, 1] for combination
    def normalize(scores):
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min < 1e-12:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
    
    n_ei = normalize(ei)
    n_ucb = normalize(ucb)
    
    # Blend EI and UCB
    # Early: weight UCB more (exploration)
    # Late: weight EI more (exploitation)
    # We can vary the weights based on progress
    # w_ucb = 1.0 - progress (starts at 1, ends at 0)
    # w_ei = progress (starts at 0, ends at 1)
    # But we always want some balance.
    # Let's use a fixed blend that leans towards the current phase.
    # Actually, since kappa already adapts, maybe a static blend is better?
    # Or adapt the blend too.
    # Let's try:
    # Early: 0.7 UCB, 0.3 EI
    # Late: 0.3 UCB, 0.7 EI
    w_ucb = 0.7 - 0.4 * progress
    w_ei = 0.3 + 0.4 * progress
    
    final_scores = w_ei * n_ei + w_ucb * n_ucb
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
# EVOLVE-BLOCK-END
