"""
Time-gated hybrid EI/PI acquisition with cost-awareness and strict deadline enforcement.

Strategy:
1. Compute standard Expected Improvement (EI) for minimization.
2. Compute Probability of Improvement (PI) for minimization.
3. Blend them using a time-based schedule:
   - Early (first 30% of time): Heavier PI weight to encourage exploration of uncertain regions.
   - Late (last 70% of time): Heavier EI weight to exploit the best known regions.
4. Divide by cost to account for the cost-aware constraint (cheaper points get higher score).
5. Enforce a hard deadline (290s) to prevent timeout, returning the current best score if time runs out.
"""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

# Global state for deadline enforcement
_start_time = None
_DEADLINE_SECONDS = 290.0  # 10s safety margin before 300s limit

def _get_elapsed_time():
    """Get elapsed time since the start of the evaluation loop."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _check_deadline():
    """Check if we are close to the deadline."""
    return _get_elapsed_time() > _DEADLINE_SECONDS

def _ei(mu, sigma, f_best, eps=1e-9):
    """
    Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _pi(mu, sigma, f_best, eps=1e-9):
    """
    Probability of Improvement for minimization.
    PI = Phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    return norm.cdf(z)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI and PI based on elapsed time.
    
    Parameters:
    mu: array of shape (2000,) - GP mean
    sigma: array of shape (2000,) - GP standard deviation
    f_best: float - best observed value so far
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline - if we're too close, return a safe default score
    # We return a simple EI/cost score which is fast and finite
    if _check_deadline():
        # Fast fallback: simple EI / cost
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        ei = _ei(np.asarray(mu, dtype=np.float64), sigma_safe, f_best)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute EI and PI
    ei = _ei(mu, sigma_safe, f_best)
    pi = _pi(mu, sigma_safe, f_best)
    
    # Time-based blending
    # Total budget is 30.0 cost units, but time is the constraint.
    # We assume the loop runs for a fixed duration.
    # Let's use elapsed time relative to a nominal total time (e.g., 250s)
    # to determine the phase.
    elapsed = _get_elapsed_time()
    nominal_total_time = 250.0  # Estimate for when the loop might end
    progress = min(elapsed / nominal_total_time, 1.0)
    
    # Weight for PI (exploration) decreases over time
    # Weight for EI (exploitation) increases over time
    # At progress=0, w_pi=0.5, w_ei=0.5
    # At progress=1, w_pi=0.1, w_ei=0.9
    w_pi = 0.5 - 0.4 * progress
    w_ei = 1.0 - w_pi
    
    # Combine EI and PI
    # Normalize PI to be on a similar scale as EI? 
    # PI is in [0,1], EI is unbounded. 
    # A common approach is to use PI as a bonus or to scale it.
    # Let's use a simple weighted sum. To make PI comparable, we can multiply by a scale factor.
    # However, since we are dividing by cost, and cost is ~1-2, the scales matter.
    # Let's try: score = (w_ei * EI + w_pi * PI * scale) / cost
    # A reasonable scale for PI might be the mean EI or a fixed value.
    # Let's use a fixed scale of 1.0 for simplicity, or adaptively.
    # Actually, EI can be very small. Let's just use the raw values.
    # To ensure PI contributes meaningfully, we might need to boost it.
    # Let's try scaling PI by the median EI if median EI > 0, else 1.
    median_ei = np.median(ei)
    if median_ei > 1e-6:
        pi_scale = median_ei
    else:
        pi_scale = 1.0
        
    combined = w_ei * ei + w_pi * pi * pi_scale
    
    # Cost-awareness: divide by cost
    score = combined / cost_safe
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
