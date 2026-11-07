import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE = 325.0  # Safety margin before 330s limit

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with diversity penalty for Bayesian optimization.
    
    Combines Expected Improvement (EI) with a diversity-aware penalty that
    discourages redundant evaluations in well-explored (low-sigma) regions.
    
    Args:
        mu: Array of GP mean predictions, shape (N,)
        sigma: Array of GP standard deviations, shape (N,)
        f_best: Scalar, best observed value so far.
        cost: Array of evaluation costs, shape (N,)
        
    Returns:
        Array of acquisition scores, shape (N,). Higher is better.
    """
    _ensure_start()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Deadline guard: if we've exceeded our safety margin, return safe defaults
    if _time_remaining() < 0.5:
        # Return a simple, fast score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-8)
        z = np.clip((f_best - mu) / sigma_safe, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-8)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive to avoid division by zero
    # Use a scale-aware floor based on the magnitude of mu
    mu_abs_scale = max(float(np.max(np.abs(mu))) if n > 0 else 1.0, 1e-3)
    sigma_floor = 1e-6 * mu_abs_scale
    sigma_safe = np.maximum(sigma, sigma_floor)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu is low (good), f_best - mu is positive.
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Numerical safety: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Diversity-aware penalty:
    # Low sigma indicates the GP is confident, likely near already-evaluated points.
    # We penalize candidates in high-confidence regions to encourage exploration
    # of unexplored areas, helping escape local optima in the Levy function.
    
    # The penalty is proportional to how "certain" the GP is (low sigma = high certainty).
    # We use an exponential decay: penalty = exp(-sigma / scale)
    # When sigma is large (uncertain/unexplored), penalty -> 0 (no penalty)
    # When sigma is small (certain/explored), penalty -> 1 (maximum penalty)
    
    # Scale the sigma for the diversity penalty relative to the problem scale
    # Use a fraction of the typical sigma range
    sigma_max = float(np.max(sigma_safe)) if n > 0 else 1.0
    sigma_scale = max(sigma_max * 0.3, 1e-6)
    
    # Diversity penalty: higher when sigma is low (well-explored region)
    # This creates a "crowding" effect that pushes search away from dense clusters
    diversity_penalty = np.exp(-sigma_safe / sigma_scale)
    
    # Weight the diversity penalty relative to EI
    # The weight should be small enough not to overwhelm EI but large enough
    # to provide meaningful diversification
    # Scale the penalty weight by the magnitude of EI to keep it balanced
    ei_scale = max(float(np.max(ei)) if n > 0 else 1.0, 1e-8)
    diversity_weight = 0.15 * ei_scale
    
    # Apply the diversity penalty
    # Subtract the penalty from EI before normalizing by cost
    ei_diverse = ei - diversity_weight * diversity_penalty
    
    # Ensure the diversified EI is non-negative
    ei_diverse = np.maximum(ei_diverse, 0.0)
    
    # Cost-aware score: diversified EI / cost
    scores = ei_diverse / cost_safe
    
    # Handle any remaining non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores
    scores = np.maximum(scores, 0.0)
    
    return scores
