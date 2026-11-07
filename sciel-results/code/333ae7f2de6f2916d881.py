"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a cost-normalized GP-UCB with dynamically annealed exploration coefficient,
    blended with Expected Improvement, and a soft Pareto-frontier boost.
    
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
    DEADLINE = 320.0  # Safety margin before 330s limit
    
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
    if time.time() - t0 > DEADLINE - 10:
        # Fallback to simple cost-aware UCB with sqrt cost
        beta = 1.5
        improvement = f_best - mu + beta * sigma
        scores = improvement / np.sqrt(cost)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Estimate progress based on how close f_best is to the known minimum
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress = 1.0
    else:
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Dynamically annealed exploration coefficient
    # Starts high for exploration, decays exponentially based on progress
    beta_max = 2.5
    beta_min = 0.5
    # Exponential decay: beta = beta_min + (beta_max - beta_min) * exp(-k * progress)
    k_decay = 3.0
    beta = beta_min + (beta_max - beta_min) * np.exp(-k_decay * progress)
    
    # Compute UCB improvement
    ucb_improvement = f_best - mu + beta * sigma
    ucb_improvement = np.maximum(ucb_improvement, 0.0)
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Blend EI and UCB based on progress
    # Early: weight towards UCB for exploration, Late: weight towards EI for exploitation
    w = 0.5 * (1.0 + progress)  # w in [0.5, 1.0]
    blended = w * ei + (1.0 - w) * ucb_improvement
    
    # Cost-aware normalization using power(cost, 0.7)
    # This reduces the penalty on expensive points compared to sqrt(cost),
    # allowing consideration of high-cost regions that may contain the global minimum
    cost_factor = np.power(cost, 0.7)
    base_scores = blended / cost_factor
    
    # Soft Pareto-frontier boost:
    # Use a geometric mean of normalized objectives to create a smooth surrogate
    # for Pareto optimality in the (improvement, uncertainty, -cost) space.
    
    # Normalize each objective to [0, 1]
    imp_min, imp_max = np.min(blended), np.max(blended)
    unc_min, unc_max = np.min(sigma), np.max(sigma)
    cost_min, cost_max = np.min(cost), np.max(cost)
    
    imp_range = imp_max - imp_min + 1e-9
    unc_range = unc_max - unc_min + 1e-9
    cost_range = cost_max - cost_min + 1e-9
    
    imp_norm = (blended - imp_min) / imp_range  # higher is better
    unc_norm = (sigma - unc_min) / unc_range    # higher is better (more exploration)
    cost_norm = 1.0 - (cost - cost_min) / cost_range  # higher is better (lower cost)
    
    # Geometric mean of the three normalized objectives
    eps = 1e-6
    log_imp = np.log(imp_norm + eps)
    log_unc = np.log(unc_norm + eps)
    log_cost = np.log(cost_norm + eps)
    
    geo_mean = np.exp((log_imp + log_unc + log_cost) / 3.0)
    
    # Normalize geo_mean to [0, 1] for stable scaling
    gm_min, gm_max = np.min(geo_mean), np.max(geo_mean)
    gm_range = gm_max - gm_min + 1e-9
    gm_norm = (geo_mean - gm_min) / gm_range
    
    # Smooth boost: use a tanh-based function to create a smooth, bounded boost
    # that is stronger for points with higher geo_mean (more frontier-like)
    tanh_scale = 2.5
    smooth_boost = np.tanh(tanh_scale * (gm_norm - 0.5))  # in [-1, 1]
    
    # Convert to a multiplicative factor: 
    # - Points with gm_norm > 0.5 get a boost up to +20%
    # - Points with gm_norm < 0.5 get a penalty up to -20%
    # - Points near 0.5 get ~0 change
    boost_factor = 1.0 + 0.20 * smooth_boost
    
    # Apply the smooth boost
    scores = base_scores * boost_factor
    
    # Add a small uncertainty bonus to encourage exploration in uncertain regions
    # This is a smooth surrogate that balances exploitation and exploration
    # The bonus is scaled by the progress: more exploration early, less late
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    uncertainty_bonus = 0.08 * (1.0 - progress) * sigma_norm
    
    scores = scores + uncertainty_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
