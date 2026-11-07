"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses an adaptive Lower Confidence Bound (LCB) strategy for minimization.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Safety margin: If the total time exceeds 300s, we return a simplified score
    # to ensure we don't risk a timeout in the evaluator's loop.
    if time.time() - _start_time > 300:
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is positive to avoid division by zero or invalid values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration coefficient:
    # Early calls (first ~10 evaluations): high exploration to cover the 6D space
    # Later calls: shift towards exploitation to refine the best region
    # Total evaluations: 30 (5 initial + 25 optimization steps)
    total_evals = 30
    progress = min(_call_count / total_evals, 1.0)
    
    # Start with high exploration (kappa=4.0) and decay to exploitation (kappa=1.5)
    # This ensures early global coverage, then focuses on the promising region
    kappa = 1.5 + 2.5 * (1.0 - progress)
    
    # LCB for minimization:
    # We want to maximize the potential for finding a low value.
    # Score = -mu + kappa * sigma
    # Higher sigma encourages exploring uncertain regions
    # Lower mu encourages exploiting known good regions
    scores = -mu + kappa * sigma_safe
    
    # Add a small noise term to break ties and encourage diversity
    # This helps escape local optima in the multimodal landscape
    rng = np.random.RandomState(_call_count)
    noise = 0.01 * rng.randn(len(mu))
    scores += noise
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
