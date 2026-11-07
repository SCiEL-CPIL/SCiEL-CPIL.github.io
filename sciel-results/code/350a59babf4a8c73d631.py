"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    Uses a time-aware ensemble of candidate acquisition functions,
    returning the best-performing score vector within the time budget.
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin before 330s kill
    
    # Ensure arrays are numpy
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Basic sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-aware Expected Improvement with bounded exploration bonus
    # EI / cost, normalized by improvement scale, plus small uncertainty bonus
    
    z1 = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with extreme values
    z1_clipped = np.clip(z1, -10, 10)
    ei1 = (f_best - mu) * norm.cdf(z1_clipped) + sigma * norm.pdf(z1_clipped)
    
    # Normalize EI by cost and by the scale of f_best to make it dimensionless
    # Use max(f_best, 1) as scale to avoid division by tiny f_best
    scale = max(abs(f_best), 1.0)
    cost_aware_ei = ei1 / (cost * scale)
    
    # Bounded exploration bonus: small weight for high uncertainty
    # This helps escape local optima in Rastrigin's many local minima
    # Use a soft saturation: alpha * sigma / (sigma + sigma_ref)
    sigma_ref = 0.1 * scale  # reference uncertainty scale
    alpha = 0.3  # exploration weight
    exploration_bonus = alpha * sigma / (sigma + sigma_ref)
    
    # Combine: cost-aware EI dominates, exploration is a small additive bonus
    final_score = cost_aware_ei + exploration_bonus
    
    # Ensure finite
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_score
# EVOLVE-BLOCK-END
