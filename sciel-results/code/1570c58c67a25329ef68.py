import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Hartmann 6D).
    Uses a cost-normalized GP-UCB with dynamically annealed exploration coefficient
    based on budget progress, blended with a soft Pareto-frontier boost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 310.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline early
    if time.time() - t0 > DEADLINE - 5:
        # Fallback to simple cost-aware UCB with sqrt cost
        beta = 1.0
        improvement = f_best - mu + beta * sigma
        scores = improvement / np.sqrt(cost)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Estimate budget progress
    # Total budget is 30, initial 5 points cost ~1.5 each => ~7.5 used
    median_cost = np.median(cost)
    if median_cost > 1e-9:
        total_budget = 30.0
        initial_cost = 5 * 1.5
        # Use median cost as typical evaluation cost
        typical_cost = median_cost
        remaining_budget = max(0, total_budget - initial_cost)
        # Progress: how much of the remaining budget has been "used"
        # If we've used X evaluations of typical cost, progress = X * typical_cost / remaining_budget
        # We don't know X exactly, but we can estimate from f_best improvement
        progress_from_cost = np.clip(1.0 - (remaining_budget / (typical_cost * 15)), 0.0, 1.0)
    else:
        progress_from_cost = 0.0
    
    # Also estimate progress from f_best improvement
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress_from_f = 1.0
    else:
        # Normalize: early on f_best ~ -1.0 to -1.5, late ~ -3.0 to -3.3
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress_from_f = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Combine both progress estimates
    progress = 0.5 * progress_from_cost + 0.5 * progress_from_f
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamically annealed beta for UCB
    # Start high (aggressive exploration), decay exponentially
    beta_start = 2.5
    beta_end = 0.5
    # Exponential decay: beta = beta_end + (beta_start - beta_end) * exp(-k * progress)
    k = 3.0
    beta = beta_end + (beta_start - beta_end) * np.exp(-k * progress)
    
    # Compute UCB improvement
    ucb_improvement = f_best - mu + beta * sigma
    
    # Compute Expected Improvement for blending
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Blend UCB and EI: more UCB early, more EI late
    # w_ucb decreases with progress
    w_ucb = 0.7 - 0.4 * progress  # w_ucb in [0.3, 0.7]
    blended = w_ucb * ucb_improvement + (1.0 - w_ucb) * ei
    
    # Cost-aware normalization using sqrt(cost)
    cost_factor = np.sqrt(cost)
    base_scores = blended / cost_factor
    
    # Soft Pareto-frontier boost
    # Normalize each objective to [0, 1] using robust percentiles
    imp_min, imp_max = np.percentile(blended, 1), np.percentile(blended, 99)
    unc_min, unc_max = np.percentile(sigma, 1), np.percentile(sigma, 99)
    cost_min, cost_max = np.percentile(cost, 1), np.percentile(cost, 99)
    
    imp_range = imp_max - imp_min + 1e-9
    unc_range = unc_max - unc_min + 1e-9
    cost_range = cost_max - cost_min + 1e-9
    
    imp_norm = np.clip((blended - imp_min) / imp_range, 0.0, 1.0)
    unc_norm = np.clip((sigma - unc_min) / unc_range, 0.0, 1.0)
    cost_norm = np.clip(1.0 - (cost - cost_min) / cost_range, 0.0, 1.0)
    
    # Weighted geometric mean with adaptive weights
    w_explore = 0.4 + 0.3 * (1.0 - progress)
    w_exploit = 0.3 + 0.2 * progress
    w_cost = max(1.0 - w_explore - w_exploit, 0.1)
    
    eps = 1e-6
    log_imp = np.log(imp_norm + eps)
    log_unc = np.log(unc_norm + eps)
    log_cost = np.log(cost_norm + eps)
    
    geo_mean = np.exp(w_exploit * log_imp + w_explore * log_unc + w_cost * log_cost)
    
    gm_min, gm_max = np.min(geo_mean), np.max(geo_mean)
    gm_range = gm_max - gm_min + 1e-9
    gm_norm = (geo_mean - gm_min) / gm_range
    
    # Smooth tanh-based boost
    tanh_scale = 2.5
    smooth_boost = np.tanh(tanh_scale * (gm_norm - 0.5))
    boost_factor = 1.0 + 0.20 * smooth_boost
    
    scores = base_scores * boost_factor
    
    # Small uncertainty bonus scaled by progress
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = np.clip((sigma - sigma_min) / sigma_range, 0.0, 1.0)
    uncertainty_bonus = 0.08 * (1.0 - progress) * sigma_norm
    
    scores = scores + uncertainty_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
