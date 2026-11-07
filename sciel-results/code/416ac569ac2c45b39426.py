"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Combines Expected Improvement with adaptive UCB exploration,
    normalized and cost-discounted for robustness.
    Uses a smooth, adaptive cost discount that balances exploitation and exploration.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive UCB exploration
    # Estimate relative uncertainty to scale exploration
    med_sigma = np.median(sigma_safe)
    std_mu = np.std(mu) + 1e-8
    # beta scales with relative uncertainty: high sigma relative to mu spread -> more exploration
    # Clip to a reasonable range to avoid extreme values
    beta = np.clip(med_sigma / std_mu, 0.2, 2.0)
    
    # UCB for minimization: -mu + beta * sigma (higher = more promising/uncertain)
    ucb = -mu + beta * sigma_safe
    
    # Normalize both components to be comparable
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-8
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Weighted combination: 65% EI, 35% UCB
    # Slight shift toward exploitation compared to inspiration 2, but keeping exploration
    score = 0.65 * ei_norm + 0.35 * ucb_norm
    
    # Cost-aware: use a smooth, adaptive cost discount
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-8
    cost_norm = (cost_safe - cost_min) / cost_range
    
    # Use a power to control aggressiveness of cost preference
    # A moderate power (1.0-1.5) balances cost awareness with exploration
    # We use a smooth discount: divide by (1 + cost_norm)^power
    # This is equivalent to multiplying by (1 + cost_norm)^(-power)
    cost_power = 1.0
    cost_discount = np.power(1.0 + cost_norm, -cost_power)
    
    # Apply cost discount to the combined score
    score = score * cost_discount
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final safety check
    if not np.all(np.isfinite(score)):
        score = np.zeros_like(score)
    
    return score
# EVOLVE-BLOCK-END
