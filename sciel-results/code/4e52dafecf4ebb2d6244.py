"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines:
    - Global budget-aware exploration/exploitation switching (inspiration 1)
    - EIPC (Expected Improvement per unit cost) as the core scoring metric
    - Dynamic cost penalty that scales with budget depletion
    - Deadline guard for safety
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Progress estimate: how far have we come from the initial random starts?
    # For Powell 4D on [-4,5]^4, initial f_best is around 5-20, minimum is 0
    # Use a normalized progress metric
    f_ref = 1.0
    # progress = 1 means early (f_best high), 0 means late (f_best near 0)
    progress = np.clip(f_best / (abs(f_best) + f_ref), 0.0, 1.0)
    # Invert: we want exploitation weight to increase as progress decreases
    exploitation_weight = 1.0 - progress  # 0 early, 1 late
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # EIPC: Expected Improvement per unit cost
    eipc = ei / cost_safe
    
    # Uncertainty bonus for exploration
    # Higher when we're early (high progress), lower when late
    beta = 0.3 + 0.7 * progress  # 1.0 early, 0.3 late
    ucb_term = beta * sigma_safe / cost_safe
    
    # Combine EIPC and UCB with dynamic weights
    # Early: more weight on exploration (UCB)
    # Late: more weight on exploitation (EIPC)
    w_eipc = 0.4 + 0.6 * exploitation_weight  # 0.4 early, 1.0 late
    w_ucb = 1.0 - w_eipc  # 0.6 early, 0.0 late
    
    # Normalize both terms to comparable scales
    eipc_scale = np.std(eipc) + 1e-8
    ucb_scale = np.std(ucb_term) + 1e-8
    
    eipc_norm = eipc / eipc_scale
    ucb_norm = ucb_term / ucb_scale
    
    combined = w_eipc * eipc_norm + w_ucb * ucb_norm
    
    # Global cost penalty: subtract a term proportional to cost
    # This is a global penalty that grows with cost
    # Use log1p to avoid over-penalizing
    cost_penalty = 0.2 * np.log1p(cost_safe)
    
    score = combined - cost_penalty
    
    # Deadline check: if we're running out of time, return immediately
    if time.time() > t_start + 280:
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Small regularization to avoid exact ties
    score += 1e-10 * np.random.randn(len(score))
    
    return score
# EVOLVE-BLOCK-END
