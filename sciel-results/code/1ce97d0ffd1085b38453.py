"""Direct cost-aware acquisition for Bayesian optimization on griewank 2d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: Cost-Adjusted Lower Confidence Bound (LCB) with dynamic kappa.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions in objective units
    - sigma: array of shape (2000,) - GP std-dev predictions in objective units
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs (1 + mean of scaled coords)
    
    Returns:
    - scores: array of shape (2000,) of finite scores, higher = more promising
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Ensure inputs are float64 arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic kappa for LCB: shifts from exploration to exploitation
    # As f_best approaches the minimum (0 for griewank), we should exploit more
    # Use a heuristic based on f_best relative to a scale
    # Griewank on [-10,10]^2 has values typically in [0, ~1] for the minimum region
    # f_best starts around 0.5-1.0 and decreases toward 0
    
    # Scale f_best to get a measure of progress
    # When f_best is large (early), explore more (higher kappa)
    # When f_best is small (late), exploit more (lower kappa)
    
    # Use exponential decay: kappa starts at ~3.0 and decreases to ~0.5
    # f_scale is a reference scale; for griewank, values near the minimum are < 0.1
    f_scale = 0.1  # Reference scale for griewank minimum region
    
    # kappa = 0.5 + 2.5 * exp(-f_best / f_scale)
    # When f_best >> f_scale: kappa ≈ 0.5 + 2.5 * 0 = 0.5 (exploit)
    # When f_best << f_scale: kappa ≈ 0.5 + 2.5 * 1 = 3.0 (explore)
    # Wait, that's backwards. Let me reconsider.
    
    # Actually:
    # Early on, f_best is large (e.g., 0.5), so f_best/f_scale = 5, exp(-5) ≈ 0.007
    # So kappa ≈ 0.5 + 2.5 * 0.007 ≈ 0.52 (too low for exploration)
    
    # Let me reverse:
    # kappa = 0.5 + 2.5 * exp(-f_best / f_scale) gives low kappa when f_best is large
    # That's wrong. I want high kappa (exploration) when f_best is large (early)
    
    # So: kappa = 0.5 + 2.5 * (1 - exp(-f_best / f_scale))
    # When f_best is large: 1 - exp(-large) ≈ 1, kappa ≈ 3.0 (explore)
    # When f_best is small: 1 - exp(-small) ≈ small, kappa ≈ 0.5 (exploit)
    
    # But f_best could be 0 or very close to 0 at the end
    # Let's use: kappa = 0.5 + 2.5 * np.exp(-f_best / f_scale)
    # No, that's still backwards.
    
    # Let me think again:
    # Early: f_best ≈ 0.5, we want to explore → high kappa
    # Late: f_best ≈ 0.01, we want to exploit → low kappa
    
    # kappa = base + range * exp(-f_best / scale)
    # Early: f_best=0.5, scale=0.1, exp(-5)≈0.007, kappa ≈ base + range*0.007
    # Late: f_best=0.01, scale=0.1, exp(-0.1)≈0.9, kappa ≈ base + range*0.9
    
    # That's still backwards. I need the opposite.
    
    # Use: kappa = base + range * (1 - exp(-f_best / scale))
    # Early: f_best=0.5, 1-exp(-5)≈0.993, kappa ≈ base + range*0.993 (high, explore)
    # Late: f_best=0.01, 1-exp(-0.1)≈0.095, kappa ≈ base + range*0.095 (low, exploit)
    
    base_kappa = 0.5
    range_kappa = 2.5
    f_scale = 0.1
    
    kappa = base_kappa + range_kappa * (1.0 - np.exp(-max(f_best, 0.0) / f_scale))
    
    # Clamp kappa to reasonable range
    kappa = np.clip(kappa, 0.3, 5.0)
    
    # LCB score: mu - kappa * sigma
    # We want to maximize this (lower is better for the objective, so higher LCB score is better)
    lcb = mu - kappa * sigma_safe
    
    # Cost-adjusted score: lcb / cost
    # Higher score means better (lower function value per unit cost)
    scores = lcb / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
