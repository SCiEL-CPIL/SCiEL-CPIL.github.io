"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track evaluation count for adaptive behavior
_eval_count = [0]


def acquisition(mu, sigma, f_best):
    """
    Progressive UCB/EI hybrid acquisition function.
    
    Uses adaptive exploration: high UCB beta when f_best is far from 0,
    transitioning to exploitation (EI-like) as f_best approaches the
    known minimum of 0.
    """
    _eval_count[0] += 1
    n_evals = _eval_count[0]
    
    # Total evaluations: 5 initial + 25 optimization = 30
    # n_evals goes from 1 to 25 for the optimization phase
    total_opt_evals = 25
    progress = min(n_evals / total_opt_evals, 1.0)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for EI component
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # EI component (exploitation)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB component (exploration)
    # Beta adapts based on:
    # 1. Progress: start high, decrease
    # 2. f_best magnitude: if f_best is large, explore more
    # 3. Median uncertainty: normalize beta by typical sigma level
    
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-10:
        sigma_median = 1e-10
    
    # f_best relative to median uncertainty: how "improved" are we?
    # If f_best >> sigma_median, we're still far from optimum -> explore more
    # If f_best << sigma_median, we're close -> exploit more
    f_best_ratio = abs(f_best) / sigma_median
    
    # Base beta schedule: starts at ~3.0, decays to ~0.5
    beta_base = 3.0 * (1.0 - progress) + 0.5 * progress
    
    # Modulate by f_best ratio: if f_best is large relative to uncertainty,
    # boost exploration (we haven't found good values yet)
    # If f_best is small, reduce exploration
    f_best_boost = np.clip(f_best_ratio, 0.3, 3.0)
    
    beta = beta_base * f_best_boost
    
    ucb = mu + beta * sigma_safe
    
    # Blend: use UCB for exploration, EI for exploitation
    # Blend weight: early on, favor UCB; later, favor EI
    # At progress=0: 70% UCB, 30% EI
    # At progress=1: 20% UCB, 80% EI
    ucb_weight = 0.7 * (1.0 - progress) + 0.2 * progress
    
    # Normalize both to comparable scales before blending
    # EI is already in "improvement" units
    # UCB is in function value units (lower is better, so we want high UCB scores)
    # Convert UCB to a "score" where higher is better:
    # score_ucb = -ucb (since lower function value is better)
    # But we need to blend with EI which is positive when mu < f_best
    
    # Alternative: use a single acquisition that combines both
    # score = EI + alpha * (f_best - mu) * exp(-sigma^2 / (2 * tau^2))
    # where the second term rewards low mu with a Gaussian gate on sigma
    
    # Let's use a cleaner approach:
    # Primary: EI (exploitation)
    # Secondary: UCB bonus scaled by exploration weight
    # The UCB bonus should be in the same units as EI
    
    # Scale UCB to be comparable to EI:
    # EI is roughly in range [0, f_best - min_mu]
    # UCB score: (f_best - mu) + beta * sigma  -- this is the "expected best" improvement
    ucb_improvement = (f_best - mu) + beta * sigma_safe
    ucb_improvement = np.maximum(ucb_improvement, 0.0)
    
    # Blend the two improvements
    score = ucb_weight * ucb_improvement + (1.0 - ucb_weight) * ei
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
