"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement per unit cost as the primary metric,
    with a small adaptive exploration bonus to balance exploitation
    and exploration while respecting evaluation costs.
    """
    # Deadline guard
    t0 = time.time()
    
    n = len(mu)
    
    # Ensure inputs are finite
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros(n)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement (EI) in objective units
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # This naturally favors cheaper evaluations with similar improvement
    cost_score = ei / cost_safe
    
    # Adaptive exploration: when EI is low everywhere, encourage exploration
    # of uncertain regions, but scaled by cost efficiency
    max_ei = np.max(ei)
    if max_ei < 1e-8:
        # Almost no expected improvement anywhere -> explore
        # Use UCB-like score divided by cost
        beta = 1.0
        ucb = -mu + beta * sigma_safe
        score = ucb / cost_safe
    else:
        # Blend: primarily cost-normalized EI, with a small UCB term
        # The UCB term is scaled to be a minor influence
        improvement = np.maximum(f_best - mu, 0.0)
        # beta decreases as improvement potential increases (more exploitation)
        beta = 0.3 * np.exp(-improvement / (sigma_safe + 1e-10))
        beta = np.clip(beta, 0.05, 0.3)
        
        ucb = -mu + beta * sigma_safe
        ucb_cost = ucb / cost_safe
        
        # Primary: cost-normalized EI; secondary: cost-normalized UCB
        score = 0.85 * cost_score + 0.15 * ucb_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we're close to deadline, return simple fallback
    elapsed = time.time() - t0
    if elapsed > 250:
        fallback = ei / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
# EVOLVE-BLOCK-END
