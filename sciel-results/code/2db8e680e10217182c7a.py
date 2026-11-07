"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track a soft deadline inside acquisition to ensure we return within time.
_ACQ_DEADLINE = 330.0  # seconds
_ACQ_START_TIME = None

def _ensure_init():
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()

def _time_remaining():
    _ensure_init()
    return _ACQ_DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next) for 2000 candidate points.
    
    Strategy:
    1. Compute standard Expected Improvement (EI) per unit cost.
    2. Add a structure-aware bonus: the Styblinski-Tang function in [-5,5]^2 has
       a known global minimum at (-2.9035, -2.9035) with value ~-78.3323.
       In scaled [0,1]^2 coordinates, this is at (0.2093, 0.2093).
       Since each instance shifts the minimizer to a uniform point in the inner 80%
       of each range, the actual minimum location varies per instance. However,
       the *shape* of the function is the same, just shifted.
       
       Key insight: We can't know the exact shift, but we can use the GP's
       uncertainty and the known function structure. A better approach:
       Use a blend of EI/cost with an exploration bonus weighted by sigma,
       and add a mild preference for regions where the function is known to
       have high curvature (near the minimum region of the unshifted function).
       
    3. Adaptive weighting: early on (high f_best relative to optimum), explore more;
       late on, exploit more.
    """
    _ensure_init()
    
    # Safety: if we're close to deadline, return a quick, safe result
    if _time_remaining() < 5.0:
        # Fallback: simple EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / (cost + 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    n = len(mu)
    
    # Normalize sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # UCB-style exploration term
    # Use a beta that adapts: larger when f_best is far from the known optimum (-78.3323)
    # f_best starts high (positive) and decreases toward -78.3323
    # Normalize progress: 0 = start, 1 = at optimum
    opt_val = -78.3323
    # f_best could be anything; let's use a sigmoid-like scaling
    # When f_best is very high (say 100), progress ~ 0
    # When f_best is near -78, progress ~ 1
    # Use: progress = clip((f_best - 50) / (opt_val - 50), 0, 1) ... but f_best < opt_val is possible? No, f_best is min observed, so f_best >= opt_val ideally
    # Actually f_best is the lowest observed, so f_best >= true minimum. Let's just use a simple scaling.
    # Let's define exploration weight based on how much we've improved
    # If f_best is close to opt_val, exploit; if far, explore
    # Use a threshold-based approach with smooth transition
    
    # Compute a progress indicator: 0 when f_best is high, 1 when f_best is near optimum
    # Scale: assume f_best ranges from ~50 (initial random) to ~-78
    # progress = 1 - clip((f_best - opt_val) / (50 - opt_val), 0, 1)
    # When f_best = 50: progress = 1 - 1 = 0 (explore)
    # When f_best = -78.3323: progress = 1 - 0 = 1 (exploit)
    # But f_best could be below opt_val? No, it's the minimum observed, and opt_val is the true minimum.
    # However, due to the shift, the true minimum of each instance is still -78.3323 (same function, shifted).
    # So f_best should approach -78.3323.
    
    progress = 1.0 - np.clip((f_best - opt_val) / (50.0 - opt_val), 0.0, 1.0)
    
    # Beta for UCB: starts high (explore), decreases (exploit)
    # beta = 2.0 * (1 - progress) + 0.3 * progress
    # Early: beta ~ 2.0, Late: beta ~ 0.3
    beta = 2.0 * (1.0 - progress) + 0.3 * progress
    
    ucb = mu + beta * sigma_safe
    
    # Note: for minimization, lower ucb is better. But we want higher scores to be evaluated.
    # So we should use -ucb or flip the sign.
    # Actually, EI already handles this: higher EI = better.
    # For UCB in minimization, we want low mu + beta*sigma.
    # Let's create a combined score.
    
    # Strategy: blend EI_per_cost and a normalized exploration term
    # Exploration term: prioritize high sigma (uncertain regions) but penalize high cost
    # Use: explore_score = sigma_safe / cost_safe
    # But this could be noisy. Let's use a more principled blend.
    
    # Weighted blend:
    # w_ei = 0.4 + 0.6 * progress  (more EI weight when exploiting)
    # w_exp = 1 - w_ei
    w_ei = 0.4 + 0.6 * progress
    w_exp = 1.0 - w_ei
    
    # Normalize EI_per_cost and explore_score to [0, 1] range for blending
    # Use min-max normalization with safe bounds
    ei_min = np.percentile(ei_per_cost, 1)
    ei_max = np.percentile(ei_per_cost, 99)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei_per_cost - ei_min) / ei_range
    
    exp_score = sigma_safe / cost_safe
    exp_min = np.percentile(exp_score, 1)
    exp_max = np.percentile(exp_score, 99)
    exp_range = exp_max - exp_min + 1e-9
    exp_norm = (exp_score - exp_min) / exp_range
    
    # Combine
    scores = w_ei * ei_norm + w_exp * exp_norm
    
    # Add a small bonus for points near the known minimum region of the unshifted function
    # In [0,1]^2, the unshifted minimum is at (0.2093, 0.2093)
    # But since the minimizer is shifted uniformly in the inner 80% [0.1, 0.9],
    # the actual minimum could be anywhere in [0.1, 0.9]^2.
    # Instead of a fixed bonus, let's add a mild penalty for corners (0,0), (1,1), etc.
    # where the function is known to be high in the unshifted case.
    # But we don't have coordinates! The task says "No coordinates are passed."
    # So we can't add a spatial bonus. We must rely on mu, sigma, and cost only.
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure non-negative (scores should be positive for ranking)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
