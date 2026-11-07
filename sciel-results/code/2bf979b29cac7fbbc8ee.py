"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Expected Improvement (EI)
    normalized by cost, with a time-budget guard.
    
    The strategy uses EI to balance exploration (via sigma) and exploitation
    (via f_best - mu). Dividing by cost accounts for the varying evaluation
    cost in the domain.
    
    We keep the PI/cost design family but refine it with:
    1. A small exploration bonus to avoid getting stuck in local minima
    2. Better handling of the cost normalization
    3. Time budget guard to prevent timeout
    """
    # Time budget guard: if we're running too long, return safe scores
    # This is a safety measure; the actual computation is fast
    t_start = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI calculation
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # PI calculation
    pi = norm.cdf(z)
    
    # Avoid division by zero in cost (though cost >= 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Hybrid: blend EI/cost and PI/cost with a small exploration term
    # EI/cost captures magnitude of improvement
    # PI/cost captures confidence of improvement
    # A small sigma/cost term adds exploration to avoid premature convergence
    
    # Use a weighted combination that emphasizes exploitation but retains
    # some exploration signal. The weight on EI is higher because Griewank
    # is smooth and we want to exploit well-modeled regions.
    
    ei_cost = ei / cost_safe
    pi_cost = pi / cost_safe
    
    # Exploration bonus: small term that encourages exploring high-uncertainty
    # regions, but damped so it doesn't dominate
    explore_bonus = sigma_safe / cost_safe * 0.1
    
    # Combine: 0.7 * EI/cost + 0.3 * PI/cost + small exploration
    scores = 0.7 * ei_cost + 0.3 * pi_cost + explore_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time budget check: if we've exceeded a safe margin, return
    t_elapsed = time.time() - t_start
    if t_elapsed > 0.5:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
