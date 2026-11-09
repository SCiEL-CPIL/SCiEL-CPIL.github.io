import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid LCB + EI strategy
    for Bayesian optimization on Shekel 4d.
    
    Combines LCB (linear exploration/exploitation) with EI (improvement-based)
    using an adaptive mixing weight that shifts from exploration to exploitation
    as the budget depletes. The cost-aware normalization accounts for the
    non-uniform evaluation cost (1 + mean(x scaled to [0,1])).
    """
    global _start_time, _call_count
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used most of the 330s budget, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 315:  # 15s safety margin
        # Return a simple, fast score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    _call_count += 1
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute EI (Expected Improvement) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute LCB (Lower Confidence Bound) for minimization
    # LCB score (higher is better) = -mu + beta * sigma
    # Adaptive beta based on call count (proxy for budget depletion)
    # Early calls: more exploration (higher beta), late calls: more exploitation
    # Budget is 30 total cost, starting with 5 points
    # Each evaluation costs 1 + mean(x) ≈ 1.5 on average
    # So roughly 15-20 evaluations total
    # Use call count as a rough proxy
    
    # Normalize call count: assume ~15 evaluations max
    # progress = 0 at start, 1 at end
    max_expected_calls = 15.0
    progress = min(1.0, max(0.0, _call_count / max_expected_calls))
    
    # Beta from 2.5 (early, exploration) to 0.3 (late, exploitation)
    beta = 2.5 - 2.2 * progress
    
    lcb_score = -mu + beta * sigma
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    # Hybrid: mix EI and LCB with adaptive weight
    # Early: more LCB (exploration), Late: more EI (exploitation of improvement)
    # w = weight on LCB, (1-w) = weight on EI
    # Start with w=0.6 (more LCB), end with w=0.2 (more EI)
    w = 0.6 - 0.4 * progress
    
    # Normalize both scores to comparable scales
    # EI is in objective units (improvement), LCB is also in objective units
    # They should be roughly comparable
    hybrid_score = w * lcb_score + (1.0 - w) * ei
    
    # Cost-aware normalization: divide by cost
    # This makes the acquisition function prefer cheaper evaluations
    score = hybrid_score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
