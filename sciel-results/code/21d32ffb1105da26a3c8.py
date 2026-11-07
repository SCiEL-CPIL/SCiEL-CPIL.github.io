# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s wall-clock
_T0 = time.time()
_DEADLINE = 330.0

def _remaining_time():
    return _DEADLINE - (time.time() - _T0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline: if less than 2s remaining, return safe fallback
    if _remaining_time() < 2.0:
        return np.ones(2000, dtype=np.float64)
    
    try:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        # Safety clamps
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        
        # Adaptive LCB strategy
        # f_best starts high (Ackley max ~ 2.5 in this domain usually, but can be higher depending on shift)
        # Min is 0.
        # We want beta to be high when f_best is far from 0, and low when f_best is close to 0.
        
        # Estimate the "scale" of the problem. 
        # Ackley 2D on [-5,5] has max value around 2.5-3.0.
        # Let's assume a reference max of 3.0 for normalization.
        # As f_best -> 0, progress -> 1. As f_best -> 3, progress -> 0.
        
        # Use a robust progress metric. 
        # If f_best is negative (shouldn't happen for Ackley min 0, but safe), clamp.
        f_best_clamped = np.clip(f_best, 0.0, 3.0)
        
        # Progress: 0 at start (f_best=3), 1 at end (f_best=0)
        progress = 1.0 - (f_best_clamped / 3.0)
        
        # Beta decays from 2.5 (exploration) to 0.2 (exploitation)
        # Using a power law or linear decay. Linear is simple and effective.
        beta = 2.5 - 2.3 * progress
        
        # Ensure beta is positive
        beta = np.maximum(beta, 0.1)
        
        # LCB: mu - beta * sigma
        # We want to maximize (f_best - LCB) / cost ? 
        # Or just maximize LCB improvement potential?
        # Standard LCB acquisition for minimization is: minimize mu - beta*sigma.
        # In a "higher is better" score framework:
        # Score = (f_best - (mu - beta*sigma)) / cost
        # This rewards points where the lower bound is significantly lower than current best.
        
        lcb = mu - beta * sigma
        scores = (f_best - lcb) / cost
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
    
    except Exception:
        # Fallback: simple EI/cost
        try:
            sigma_safe = np.maximum(sigma, 1e-9)
            cost_safe = np.maximum(cost, 1e-9)
            z = (f_best - mu) / sigma_safe
            ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
            scores = ei / cost_safe
            scores = np.where(np.isfinite(scores), scores, 0.0)
            return scores
        except Exception:
            return np.ones(2000, dtype=np.float64)
# EVOLVE-BLOCK-END
