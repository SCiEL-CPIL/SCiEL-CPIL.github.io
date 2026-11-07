"""Cost-aware acquisition function combining EI with adaptive exploration."""
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
    Cost-aware acquisition: EI/cost + adaptive exploration (sqrt-sigma + UCB).
    Exploration weight increases as EI converges to escape local optima.
    """
    _get_start_time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 315.0:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_per_cost = ei / cost_safe
    
    # Normalize EI to [0, 1]
    ei_scale = np.max(ei_per_cost)
    if ei_scale > 0:
        ei_norm = ei_per_cost / ei_scale
    else:
        ei_norm = np.zeros_like(ei_per_cost)
    
    # --- Adaptive exploration ---
    # Convergence indicator: how small is the max EI relative to f_best?
    # When EI is tiny, the GP is confident -> increase exploration
    # Use mean of top-10% EI as a robust convergence signal
    n = len(ei_norm)
    top_k = max(1, n // 10)
    top_ei = np.sort(ei_norm)[-top_k:]
    ei_convergence = np.mean(top_ei)  # close to 0 => converged
    
    # Exploration weight: base 0.1, ramps up to 0.5 as EI converges
    # Smooth ramp: weight = base + (max_weight - base) * (1 - ei_convergence)^2
    base_w = 0.10
    max_w = 0.50
    exploration_weight = base_w + (max_w - base_w) * (1.0 - min(ei_convergence, 1.0)) ** 2
    
    # Sqrt-scaled exploration: sqrt(sigma) grows slower than sigma,
    # avoiding over-exploration in high-uncertainty regions
    sigma_sqrt = np.sqrt(sigma_safe)
    sigma_sqrt_max = np.max(sigma_sqrt)
    if sigma_sqrt_max > 0:
        sigma_sqrt_norm = sigma_sqrt / sigma_sqrt_max
    else:
        sigma_sqrt_norm = np.zeros_like(sigma_sqrt)
    
    # UCB term: mu + beta * sigma (penalizes high mean, rewards uncertainty)
    # For minimization: score should be high when mu is low AND sigma is high
    # UCB_exploration = -mu_norm + beta * sigma_norm
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-12:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)  # 0=best, 1=worst
    else:
        mu_norm = np.zeros_like(mu)
    
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # UCB exploration component: reward low mean + high uncertainty
    ucb_beta = 1.0
    ucb_score = (1.0 - mu_norm) + ucb_beta * sigma_norm
    
    # Blend sqrt-sigma exploration and UCB
    # Weight 60% sqrt-sigma, 40% UCB
    exploration_score = 0.6 * sigma_sqrt_norm + 0.4 * ucb_score
    
    scores = ei_norm + exploration_weight * exploration_score
    
    # Final safety
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
