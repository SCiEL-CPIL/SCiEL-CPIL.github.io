"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite and positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB with adaptive beta: start exploratory, become more exploitative
    # beta scales with remaining budget relative to total
    # cost ranges from ~1 to ~2, budget is 30
    # Use a simple heuristic: beta decreases as f_best improves from initial
    # Since we don't know initial f_best, use a fixed moderate exploration
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Combine: weighted blend of EI and UCB, both divided by cost
    # EI captures expected gain, UCB captures potential
    # Normalize by cost to prefer cheaper points
    score_ei = ei / cost
    score_ucb = ucb / cost
    
    # Blend: early on favor UCB (exploration), later favor EI (exploitation)
    # Since we don't have iteration count, use a fixed blend
    # The cost-awareness already helps: expensive points are penalized
    score = 0.5 * score_ei + 0.5 * score_ucb
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
