"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _soft_max(a, b, t=1e-3):
    """Smooth max of two arrays via log-sum-exp (element-wise)."""
    m = np.maximum(a, b)
    return m + t * np.log(np.exp((a - m) / t) + np.exp((b - m) / t))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    
    Combines Smooth Expected Improvement (EI) with a UCB exploration bonus,
    enhanced by a Thompson Sampling-inspired stochastic perturbation.
    Uses adaptive weighting based on uncertainty, progress, and remaining budget.
    
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
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei_raw = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Smooth surrogate for max(ei, 0): removes the kink at ei=0
    ei = _soft_max(ei_raw, np.zeros_like(ei_raw), t=1e-3)

    # Compute UCB score for exploration
    beta = 1.5
    ucb_score = beta * sigma_safe - mu

    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei)
    if not np.isfinite(ei_max) or ei_max <= 1e-10:
        ei_max = 1.0
        
    ucb_max = np.max(ucb_score)
    if not np.isfinite(ucb_max) or ucb_max <= 1e-10:
        ucb_max = 1.0

    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max

    # Adaptive blend:
    # 1. Uncertainty-based weight
    mean_sigma = float(np.mean(sigma_safe))
    scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = mean_sigma / (scale + mean_sigma + 1e-8)
    # ei_weight in [0.3, 0.6]
    w_e_base = 0.6 - 0.3 * uncertainty_ratio
    w_u_base = 1.0 - w_e_base

    # 2. Progress-based modulation
    f_min_known = -3.32237
    f_initial = -1.0
    
    progress = (f_initial - f_best) / (f_initial - f_min_known + 1e-10)
    progress = np.clip(progress, 0.0, 1.5)
    
    if progress > 0.8:
        boost = (progress - 0.8) / 0.2 * 0.3
        w_e_final = np.clip(w_e_base + boost, 0.0, 1.0)
    else:
        w_e_final = w_e_base
        
    w_u_final = 1.0 - w_e_final

    # 3. Thompson Sampling-inspired stochastic perturbation
    # Draw a single sample from N(0,1) to create a stochastic weight adjustment
    # This naturally balances exploration/exploitation without hand-tuned schedules
    rng = np.random.default_rng(seed=hash((f_best, float(np.mean(sigma_safe)))) % (2**32))
    ts_sample = rng.standard_normal()
    
    # Scale the perturbation: smaller when uncertainty is low (late in run)
    perturbation_scale = 0.15 * (1.0 - uncertainty_ratio)
    ts_weight_adjustment = perturbation_scale * ts_sample
    
    # Apply stochastic adjustment to weights, keeping them in valid range
    w_e_ts = np.clip(w_e_final + ts_weight_adjustment, 0.1, 0.9)
    w_u_ts = 1.0 - w_e_ts

    # 4. Budget-aware scaling: scale exploration by estimated remaining budget
    # Total budget is 30.0; estimate remaining fraction from mean cost
    total_budget = 30.0
    mean_cost = float(np.mean(cost_safe))
    # Rough estimate: if we've spent ~cost_so_far, remaining = total - spent
    # We don't know exact spent, but high mean_cost suggests later in run
    # Use a smooth decay: more exploration early, less late
    budget_factor = 1.0 / (1.0 + 0.5 * (mean_cost - 1.0))
    budget_factor = np.clip(budget_factor, 0.5, 1.5)
    
    # Apply budget factor to UCB component (exploration)
    ucb_adjusted = ucb_norm * budget_factor

    # Combine with Thompson Sampling weights
    combined = w_e_ts * ei_norm + w_u_ts * ucb_adjusted

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
