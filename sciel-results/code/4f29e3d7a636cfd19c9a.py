"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None
_DEADLINE = 315.0  # 330s limit, leave 15s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value
    cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the limit, return a quick fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: simple LCB-like score
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        fallback = (f_best - mu - 1.0 * sigma_safe / np.sqrt(cost_safe))
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-Adjusted Lower Confidence Bound (LCB) for minimization
    # We want to minimize, so lower is better.
    # LCB = mu - kappa * sigma / sqrt(cost)
    # We want to maximize the score, so we negate it or use a different form.
    # Actually, standard LCB for minimization is: mu - kappa * sigma
    # We want to pick the point with the lowest expected value.
    # So the "score" to maximize should be related to -LCB or similar.
    # Let's define the acquisition function as:
    # score = -mu + kappa * sigma / sqrt(cost)
    # This encourages low mu (good) and high sigma (uncertainty) but penalizes high cost via sqrt.
    
    # We can also include an Expected Improvement (EI) term for exploitation.
    # Let's try a combination of LCB and EI, both cost-adjusted.
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: EI per unit cost (or per sqrt cost)
    # Using sqrt(cost) in denominator as per the strategy
    ei_cost = ei / np.sqrt(cost_safe)
    
    # LCB term: -mu + kappa * sigma / sqrt(cost)
    # Let's use a fixed kappa, say 2.0
    kappa = 2.0
    lcb_term = -mu + kappa * sigma_safe / np.sqrt(cost_safe)
    
    # Normalize both terms using median/MAD for robustness
    def median_mad_norm(arr):
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-10:
            mad = 1e-10
        return (arr - med) / (1.4826 * mad)  # 1.4826 scales MAD to std
    
    ei_cost_norm = median_mad_norm(ei_cost)
    lcb_norm = median_mad_norm(lcb_term)
    
    # Combine with weighted sum
    # Weight EI more for exploitation, LCB for exploration
    w_ei = 0.6
    w_lcb = 0.4
    
    scores = w_ei * ei_cost_norm + w_lcb * lcb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
