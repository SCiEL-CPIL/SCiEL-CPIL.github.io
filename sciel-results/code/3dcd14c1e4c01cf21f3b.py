"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Pareto front-based selection: computes non-dominated set with respect to
    maximizing Expected Improvement (EI) and minimizing acquisition cost,
    then scores candidates based on their rank within this front.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float
    cost: array of shape (2000,)
    
    Returns:
    score: array of shape (2000,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    # EI = E[max(f_best - f(x), 0)]
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: we want to maximize EI and minimize cost
    # For Pareto dominance: candidate A dominates B if A has higher EI AND lower cost
    # (or equal in one and better in other)
    
    n = len(mu)
    
    # Normalize EI and cost for comparison
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    
    # Handle edge cases
    if ei_max - ei_min < 1e-12:
        ei_max = ei_min + 1e-12
    if cost_max - cost_min < 1e-12:
        cost_max = cost_min + 1e-12
    
    ei_norm = (ei - ei_min) / (ei_max - ei_min)  # Higher is better
    cost_norm = (cost - cost_min) / (cost_max - cost_min)  # Lower is better
    
    # Find non-dominated set using vectorized approach
    # A candidate is dominated if there exists another with ei_norm >= its ei_norm AND cost_norm <= its cost_norm
    # (strictly better in at least one)
    
    # For efficiency, use a grid-based approximation or sort-based approach
    # Sort by EI descending, then for each candidate, check if any earlier candidate has lower cost
    
    # Sort indices by EI descending
    ei_order = np.argsort(-ei_norm)
    
    # Track the minimum cost seen so far (in EI-descending order)
    min_cost_so_far = np.full(n, cost_max + 1.0)
    
    # Process in EI-descending order
    running_min_cost = cost_max + 1.0
    non_dominated_mask = np.zeros(n, dtype=bool)
    
    for idx in ei_order:
        c = cost_norm[idx]
        if c < running_min_cost:
            non_dominated_mask[idx] = True
            running_min_cost = c
    
    # Assign scores: non-dominated points get high scores, dominated get lower
    # Within non-dominated, rank by a weighted combination for tie-breaking
    # Use hypervolume-like contribution approximation: score by EI_norm * (1 - cost_norm)
    
    pareto_score = ei_norm * (1.0 - cost_norm)
    
    # Boost non-dominated points significantly
    score = np.where(non_dominated_mask, 10.0 + pareto_score, pareto_score * 0.1)
    
    # Add a small exploration bonus for high uncertainty among non-dominated
    sigma_norm = sigma_safe / np.max(sigma_safe)
    score += non_dominated_mask * 0.5 * sigma_norm
    
    # Sanitize
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
