"""Cost-aware acquisition function using Probability-weighted Improvement for Bayesian optimization."""
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # seconds, safety margin before 330s limit


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability-weighted Improvement (PoI-based)
    combined with adaptive exploration for Bayesian optimization of cosine8 8d.
    
    The core idea: instead of Expected Improvement (which can be dominated by high-sigma
    regions with uncertain improvement), we use P(mu < f_best) * (f_best - mu) / cost,
    which directly maximizes expected regret reduction per unit cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fast score
    if _remaining_time() < 3.0:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -15, 15)
        poi = norm.cdf(z)
        improvement = np.maximum(f_best - mu, 0.0)
        cost_safe = np.maximum(cost, 1e-8)
        scores = poi * improvement / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Expected improvement magnitude (capped at 0)
    improvement = np.maximum(f_best - mu, 0.0)
    
    # Probability-weighted improvement: E[improvement | improvement > 0] * P(improvement > 0)
    # This is actually just the expected improvement, but computed differently:
    # PoI * improvement is a proxy that down-weights high-sigma uncertain regions
    # compared to raw EI which includes sigma * phi(z) term
    poi_weighted = poi * improvement
    
    # Cost-aware: improvement per unit cost
    poi_cost = poi_weighted / cost_safe
    
    # Exploration component: pure uncertainty reward, cost-normalized
    # This helps escape local minima in multimodal functions like cosine8
    exp_score = sigma_safe / cost_safe
    
    # Normalize each component to zero-mean, unit-variance for comparable scales
    poi_mean = np.mean(poi_cost)
    poi_std = np.std(poi_cost)
    if poi_std < 1e-10:
        poi_std = 1e-10
    poi_norm = (poi_cost - poi_mean) / poi_std
    
    exp_mean = np.mean(exp_score)
    exp_std = np.std(exp_score)
    if exp_std < 1e-10:
        exp_std = 1e-10
    exp_norm = (exp_score - exp_mean) / exp_std
    
    # Adaptive exploration weight based on progress
    # cosine8 minimum is -0.8, maximum is ~1.0
    min_possible = -0.8
    max_possible = 1.0
    progress = (f_best - min_possible) / (max_possible - min_possible + 1e-10)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Early on (progress near 1), explore more; late on (progress near 0), exploit more
    # Use a nonlinear schedule: more aggressive early exploration
    exploration_weight = 0.25 + 0.25 * (progress ** 1.5)
    exploration_weight = np.clip(exploration_weight, 0.2, 0.5)
    
    # Combine: primarily exploitation (PoI-weighted), with adaptive exploration
    combined = (1.0 - exploration_weight) * poi_norm + exploration_weight * exp_norm
    
    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
