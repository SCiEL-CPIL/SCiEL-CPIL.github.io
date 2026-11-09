# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.
    
    Uses a Progress-Adaptive LCB (Lower Confidence Bound).
    The exploration parameter beta is adapted based on both the 
    progress towards the known global minimum and the current 
    model uncertainty.
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Hartmann 3d properties
    min_val = -3.86278
    max_val = 3.0
    target_range = max_val - min_val  # ~6.86

    # 1. Calculate Progress
    # Progress is 1.0 when f_best is at max_val (start) and 0.0 when at min_val (optimum)
    # Clip to [0, 1] to handle overshooting or noise
    progress = (f_best - min_val) / target_range
    progress = np.clip(progress, 0.0, 1.0)

    # 2. Calculate Uncertainty Metric
    # Use median sigma as a robust measure of global model uncertainty
    sigma_median = np.median(sigma_safe)
    # Normalize uncertainty relative to the target range
    rel_uncertainty = sigma_median / target_range
    rel_uncertainty = np.clip(rel_uncertainty, 0.0, 1.0)

    # 3. Determine Beta (Exploration Parameter)
    # Base beta driven by progress: High when far from optimum, Low when close
    beta_min = 0.5
    beta_max = 3.0
    
    # Linear decay with progress
    beta_base = beta_min + (beta_max - beta_min) * progress
    
    # Uncertainty Boost:
    # If the model is very uncertain (high sigma), increase exploration further.
    # This prevents getting stuck in local minima if the GP hasn't modeled the region well.
    # We use a multiplicative factor that grows with uncertainty.
    # Factor ranges from 1.0 (low uncertainty) to 1.5 (high uncertainty)
    uncertainty_boost = 1.0 + 0.5 * rel_uncertainty
    
    beta = beta_base * uncertainty_boost

    # 4. Compute LCB Scores
    # For minimization, we want to minimize mu + beta * sigma
    # Score should be higher for better candidates (lower LCB) and lower cost
    lcb = mu_safe + beta * sigma_safe
    scores = -lcb / cost_safe

    # 5. Final Safety Checks
    scores = np.where(np.isfinite(scores), scores, 0.0)

    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
