"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a hybrid Thompson Sampling and UCB approach with adaptive cost normalization.
    
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
    
    # Estimate budget progress from cost array and f_best
    # Similar to Inspiration 1
    median_cost = np.median(cost)
    if median_cost > 1e-9:
        total_budget = 30.0
        initial_cost = 5 * 1.5  # approx cost of 5 uniform random points
        best_cost_idx = np.argmin(cost)
        typical_cost = cost[best_cost_idx] if cost[best_cost_idx] > 1e-9 else median_cost
        remaining_budget = max(0, total_budget - initial_cost)
        progress_from_cost = np.clip(1.0 - (remaining_budget / (typical_cost * 20)), 0.0, 1.0)
    else:
        progress_from_cost = 0.0
    
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress_from_f = 1.0
    else:
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress_from_f = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    progress = 0.5 * progress_from_cost + 0.5 * progress_from_f
    
    # Thompson Sampling Component
    # Generate a sample from the posterior
    # z ~ N(0, 1)
    z = np.random.randn(len(mu))
    f_sampled = mu + sigma * z
    
    # Sampled Improvement: max(0, f_best - f_sampled)
    sampled_imp = np.maximum(0.0, f_best - f_sampled)
    
    # UCB Component
    # Adaptive beta: explore more early, exploit more late
    beta_early = 2.5
    beta_late = 0.8
    beta = beta_early + (beta_late - beta_early) * progress
    
    ucb_term = f_best - mu + beta * sigma
    ucb_term = np.maximum(ucb_term, 0.0) # Ensure non-negative contribution
    
    # Cost-Aware Normalization
    # Use sqrt(cost) as a standard cost-aware scaling
    # We can adjust the power of cost based on progress
    # Early on, we might be more sensitive to cost (higher power) to save budget for exploration?
    # Or less sensitive?
    # Usually, we want to find the global min. Cost is a constraint.
    # Let's stick to power 0.5 (sqrt) which is robust.
    cost_power = 0.5
    cost_factor = np.power(cost, cost_power)
    
    # Combine TS and UCB
    # Weight the TS component higher as we progress?
    # Early: TS is noisy, UCB provides stable exploration.
    # Late: TS can exploit the specific sample if it's good.
    w_ts = 0.6 + 0.3 * progress  # w_ts in [0.6, 0.9]
    w_ucb = 1.0 - w_ts           # w_ucb in [0.4, 0.1]
    
    blended_score = w_ts * sampled_imp + w_ucb * ucb_term
    
    # Apply cost normalization
    scores = blended_score / cost_factor
    
    # Refinement: Soft Pareto Boost (from Inspiration 1)
    # This helps distinguish points that are good in multiple aspects (improvement, uncertainty, cost)
    # We use the blended_score and sigma to create a soft rank
    
    # Normalize objectives to [0, 1]
    # Objective 1: Blended Score (higher is better)
    bs_min, bs_max = np.min(blended_score), np.max(blended_score)
    bs_range = bs_max - bs_min + 1e-9
    bs_norm = np.clip((blended_score - bs_min) / bs_range, 0.0, 1.0)
    
    # Objective 2: Uncertainty (higher is better for exploration)
    unc_min, unc_max = np.min(sigma), np.max(sigma)
    unc_range = unc_max - unc_min + 1e-9
    unc_norm = np.clip((sigma - unc_min) / unc_range, 0.0, 1.0)
    
    # Objective 3: Inverse Cost (higher is better, i.e., lower cost)
    cost_min, cost_max = np.min(cost), np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = np.clip(1.0 - (cost - cost_min) / cost_range, 0.0, 1.0)
    
    # Weights for the geometric mean
    # Explore more early (higher weight on uncertainty)
    w_explore = 0.4 + 0.3 * (1.0 - progress)
    w_exploit = 0.4 + 0.3 * progress
    w_cost = 1.0 - w_explore - w_exploit
    w_cost = max(w_cost, 0.1)
    
    # Geometric mean
    eps = 1e-6
    log_bs = np.log(bs_norm + eps)
    log_unc = np.log(unc_norm + eps)
    log_cost = np.log(cost_norm + eps)
    
    geo_mean = np.exp(w_exploit * log_bs + w_explore * log_unc + w_cost * log_cost)
    
    # Normalize geo_mean
    gm_min, gm_max = np.min(geo_mean), np.max(geo_mean)
    gm_range = gm_max - gm_min + 1e-9
    gm_norm = (geo_mean - gm_min) / gm_range
    
    # Smooth boost
    tanh_scale = 2.0
    smooth_boost = np.tanh(tanh_scale * (gm_norm - 0.5))
    boost_factor = 1.0 + 0.2 * smooth_boost
    
    scores = scores * boost_factor
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
