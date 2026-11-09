"""Cost-aware acquisition using hybrid UCB/EI with cost-aware exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s wall-clock safety margin (limit is 330s)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative values
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Compute EI in a numerically stable way
    z_clipped = np.clip(z, -30.0, 30.0)
    ei = diff * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB component: rewards uncertainty
    # Use adaptive beta that balances exploration/exploitation
    # beta scales with the spread of sigma to adapt to problem scale
    sigma_mean = np.mean(sigma_safe)
    beta = 2.0 * max(sigma_mean, 1e-6)
    
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware normalization: divide by cost to get value per unit cost
    ei_score = ei / cost_safe
    ucb_score = ucb / cost_safe
    
    # Hybrid: combine EI and UCB with a weighting that favors UCB early
    # and EI later. Since we don't know the iteration, use a fixed blend.
    # The UCB term helps explore high-uncertainty regions, while EI exploits.
    # Weight: 0.5 * EI + 0.5 * UCB, but scale UCB to be comparable to EI
    # Normalize both to [0, 1] range relative to their own max
    ei_max = np.max(ei_score) if np.max(ei_score) > 0 else 1.0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 0 else 1.0
    
    ei_norm = ei_score / ei_max
    ucb_norm = ucb_score / ucb_max
    
    # Blend: 40% EI, 60% UCB to favor exploration in cost-aware setting
    score = 0.4 * ei_norm + 0.6 * ucb_norm
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
