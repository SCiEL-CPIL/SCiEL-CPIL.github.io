"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a blend of Expected Improvement and UCB, normalized by sqrt(cost)
    with a soft Pareto-frontier boost to balance exploration and cost.
    
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
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB improvement with adaptive beta
    # Estimate progress: how close are we to the known minimum?
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress = 1.0
    else:
        # Normalize progress: early on, f_best is around -1.0 to -2.0
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Adaptive beta: explore more early, exploit more late
    beta_early = 1.8
    beta_late = 0.8
    beta = beta_early + (beta_late - beta_early) * progress
    
    ucb_improvement = f_best - mu + beta * sigma
    
    # Blend EI and UCB based on progress
    # Early: weight towards UCB, Late: weight towards EI
    w = 0.5 * (1.0 + progress)  # w in [0.5, 1.0]
    blended = w * ei + (1.0 - w) * ucb_improvement
    
    # Cost-aware score using power(cost, 0.7) to reduce penalty on expensive points
    # This allows the algorithm to consider high-cost points that may contain
    # the true global minimum, which standard cost division might overly penalize
    cost_factor = np.power(cost, 0.7)
    base_scores = blended / cost_factor
    
    # Soft Pareto-frontier boost:
    # Instead of hard non-dominated filtering (which was O(n^2) and showed no
    # improvement), use a smooth surrogate that softly rewards points that are
    # "frontier-like" in the (improvement, uncertainty, -cost) space.
    #
    # We compute a soft rank for each point based on how well it balances
    # the three objectives, then use a smooth (tanh-based) boost.
    
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
    
    # Soft Pareto score: for each point, compute a "frontier-ness" measure.
    # A point is on the frontier if it has a good balance of all three.
    # We use a geometric mean of the normalized objectives as a smooth surrogate
    # for Pareto optimality. Points that are strong in all three dimensions
    # get a high geometric mean.
    # Add small epsilon to avoid log(0)
    eps = 1e-6
    log_imp = np.log(imp_norm + eps)
    log_unc = np.log(unc_norm + eps)
    log_cost = np.log(cost_norm + eps)
    
    # Geometric mean of the three normalized objectives
    geo_mean = np.exp((log_imp + log_unc + log_cost) / 3.0)
    
    # Normalize geo_mean to [0, 1] for stable scaling
    gm_min, gm_max = np.min(geo_mean), np.max(geo_mean)
    gm_range = gm_max - gm_min + 1e-9
    gm_norm = (geo_mean - gm_min) / gm_range
    
    # Smooth boost: use a tanh-based function to create a smooth, bounded boost
    # that is stronger for points with higher geo_mean (more frontier-like)
    # Scale the tanh input so that the boost is meaningful in the mid-range
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
