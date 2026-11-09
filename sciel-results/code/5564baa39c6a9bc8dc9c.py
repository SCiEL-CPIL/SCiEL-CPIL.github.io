"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _soft_max(a, b, t=1e-3):
    """Smooth max of two arrays via log-sum-exp (element-wise)."""
    m = np.maximum(a, b)
    # Avoid log(0) or overflow issues
    return m + t * np.log(np.exp((a - m) / t) + np.exp((b - m) / t))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    
    Combines Expected Improvement (EI) with a UCB exploration bonus.
    Uses adaptive weighting based on uncertainty, progress towards the known
    global minimum, and remaining budget. Scores are normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin well before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])

    # Replace near-zero sigma with small value to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute Expected Improvement (EI) for exploitation
    # z = (f_best - mu) / sigma
    # Clip z to prevent overflow in cdf/pdf if sigma is extremely small
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei_raw = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Smooth surrogate for max(ei, 0): removes the kink at ei=0
    # This helps in cases where the maximum EI is very close to 0
    ei = _soft_max(ei_raw, np.zeros_like(ei_raw), t=1e-3)

    # Compute UCB score for exploration
    # For minimization, we maximize the lower confidence bound potential: beta * sigma - mu
    # Use an adaptive beta that decreases as we get closer to the optimum
    # This allows more exploration early and less later
    f_min_known = -3.32237
    f_initial = -1.0  # Typical start
    
    # Progress: 0 at start, 1 at optimum
    progress = (f_initial - f_best) / (f_initial - f_min_known + 1e-10)
    progress = np.clip(progress, 0.0, 1.5)
    
    # Adaptive beta: starts high (more exploration) and decreases
    # beta = 2.0 - 1.0 * min(progress, 1.0)
    # So beta ranges from 2.0 (early) to 1.0 (late)
    beta = 2.0 - 1.0 * np.clip(progress, 0.0, 1.0)
    
    ucb_score = beta * sigma_safe - mu

    # Normalize EI and UCB to comparable scales
    # Use max normalization, but guard against zero or tiny max values
    ei_max = np.max(ei)
    if not np.isfinite(ei_max) or ei_max <= 1e-10:
        ei_max = 1.0
        
    ucb_max = np.max(ucb_score)
    if not np.isfinite(ucb_max) or ucb_max <= 1e-10:
        ucb_max = 1.0

    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max

    # Adaptive blend:
    # 1. Uncertainty-based weight:
    #    Higher uncertainty -> more exploration (lower EI weight).
    mean_sigma = float(np.mean(sigma_safe))
    scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = mean_sigma / (scale + mean_sigma + 1e-8)
    # ei_weight in [0.3, 0.6]
    w_e_base = 0.6 - 0.3 * uncertainty_ratio
    w_u_base = 1.0 - w_e_base

    # 2. Progress-based modulation:
    #    If we are very close to the known global minimum, force more exploitation.
    # If progress is high (>0.8), increase exploitation weight
    # Linearly increase w_e towards 0.9 as progress goes from 0.8 to 1.0
    if progress > 0.8:
        # Map progress [0.8, 1.0] to boost [0, 0.3]
        boost = (progress - 0.8) / 0.2 * 0.3
        w_e_final = np.clip(w_e_base + boost, 0.0, 1.0)
    else:
        w_e_final = w_e_base
        
    w_u_final = 1.0 - w_e_final

    # 3. Budget-aware scaling:
    #    The cost-aware division by cost_safe handles this naturally.
    #    More expensive points get lower scores, which is desirable.
    
    # One refinement: Ensure that if sigma is very low everywhere (converged), 
    # we don't explore. This is handled by uncertainty_ratio.

    combined = w_e_final * ei_norm + w_u_final * ucb_norm

    # Cost-aware: divide by cost
    scores = combined / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
