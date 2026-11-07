"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    with adaptive exploration bonus.
    
    POI provides a sharper, more decisive selection signal than EI,
    allowing the search to commit to promising low-cost regions faster.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    t_start = _get_start_time()
    
    # Deadline guard: if we're running low on time, return a simple score
    elapsed = time.time() - t_start
    if elapsed > 325.0:  # 5 second safety margin before 330s kill
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        poi = norm.cdf(z)
        return np.where(np.isfinite(poi), poi, 0.0)
    
    # Ensure inputs are valid arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # 1. Compute Probability of Improvement (POI)
    # POI = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    poi = norm.cdf(z)
    
    # 2. Cost-aware POI
    poi_per_cost = poi / cost_safe
    
    # 3. Normalize POI to [0, 1] for blending
    poi_max = np.max(poi_per_cost)
    if poi_max > 1e-12:
        poi_norm = poi_per_cost / poi_max
    else:
        poi_norm = np.zeros_like(poi_per_cost)
    
    # 4. Adaptive Exploration Weight
    # Estimate convergence: if the top 10% of POI values are very small, we are converged.
    n = len(poi_norm)
    top_k = max(1, int(n * 0.1))
    top_poi = np.sort(poi_norm)[-top_k:]
    poi_convergence = np.mean(top_poi)
    
    # Weight ramps from 0.1 (early, high POI) to 0.6 (late, low POI)
    base_w = 0.10
    max_w = 0.60
    # Smooth transition: weight = base + (max - base) * (1 - convergence)^2
    exploration_weight = base_w + (max_w - base_w) * (1.0 - min(poi_convergence, 1.0)) ** 2
    
    # 5. Exploration Score Components
    # a) Scaled Uncertainty: sqrt(sigma) normalized
    sigma_sqrt = np.sqrt(sigma_safe)
    sigma_sqrt_max = np.max(sigma_sqrt)
    if sigma_sqrt_max > 1e-12:
        sigma_sqrt_norm = sigma_sqrt / sigma_sqrt_max
    else:
        sigma_sqrt_norm = np.zeros_like(sigma_sqrt)
        
    # b) UCB-like term: Reward low mean (good prediction) and high uncertainty
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-12:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
    else:
        mu_norm = np.zeros_like(mu)
        
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
        
    # UCB score: (1 - mu_norm) is high when mu is low. + sigma_norm rewards uncertainty.
    ucb_score = (1.0 - mu_norm) + 1.0 * sigma_norm
    
    # Blend exploration components: 60% uncertainty, 40% UCB
    exploration_score = 0.6 * sigma_sqrt_norm + 0.4 * ucb_score
    
    # 6. Final Score
    scores = poi_norm + exploration_weight * exploration_score
    
    # Final safety: ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
