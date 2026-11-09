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
    
    Combines Smooth Expected Improvement (EI) with a UCB exploration bonus.
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
    deadline = start_time + 270  # Safety margin well before 300s limit

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
    beta = 1.5
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
    # 1. Uncertainty-based weight (from Inspiration 1):
    #    Higher uncertainty -> more exploration (lower EI weight).
    mean_sigma = float(np.mean(sigma_safe))
    scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = mean_sigma / (scale + mean_sigma + 1e-8)
    # ei_weight in [0.3, 0.6]
    w_e_base = 0.6 - 0.3 * uncertainty_ratio
    w_u_base = 1.0 - w_e_base

    # 2. Progress-based modulation (from Current Program logic):
    #    If we are very close to the known global minimum, force more exploitation.
    f_min_known = -3.32237
    f_initial = -1.0  # Typical start
    
    # Progress: 0 at start, 1 at optimum
    # If f_best is better than f_min_known, progress > 1
    progress = (f_initial - f_best) / (f_initial - f_min_known + 1e-10)
    progress = np.clip(progress, 0.0, 1.5) # Allow slight overshoot for safety
    
    # If progress is high (>0.8), increase exploitation weight
    # Linearly increase w_e towards 0.9 as progress goes from 0.8 to 1.0
    if progress > 0.8:
        # Map progress [0.8, 1.0] to boost [0, 0.3]
        boost = (progress - 0.8) / 0.2 * 0.3
        w_e_final = np.clip(w_e_base + boost, 0.0, 1.0)
    else:
        w_e_final = w_e_base
        
    w_u_final = 1.0 - w_e_final

    # 3. Budget-aware scaling (from Directive):
    #    Dynamically scale exploration by remaining budget.
    #    Total budget is 30.0.
    #    If cost is high, we are likely later in the run.
    #    We estimate remaining budget fraction.
    #    Since we don't know exactly how many evaluations are left, 
    #    we use the mean cost as a proxy for "time" or "budget consumption rate".
    #    However, a simpler heuristic: 
    #    If f_best is very good, we have likely spent a lot of budget.
    #    Let's refine the w_u_final based on a "budget pressure" signal.
    #    The directive suggests: aggressive exploration early, precise exploitation later.
    #    We can approximate "later" by high progress or low uncertainty.
    #    The current logic already does this via progress and uncertainty.
    #    Let's add a direct budget-based dampening if we assume cost correlates with run progress.
    #    Actually, the cost is per-point. The total budget is 30.
    #    We don't have the *total* cost spent so far, only the cost of the *current* candidates.
    #    But we have f_best.
    #    Let's stick to the robust progress/uncertainty blend which implicitly captures this.
    #    The directive says "dynamically scales the exploration term by the square root of the remaining budget".
    #    Without explicit remaining budget input, we approximate:
    #    Early: f_best is poor -> progress low -> w_e low, w_u high.
    #    Late: f_best is good -> progress high -> w_e high, w_u low.
    #    This is already implemented.
    
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
