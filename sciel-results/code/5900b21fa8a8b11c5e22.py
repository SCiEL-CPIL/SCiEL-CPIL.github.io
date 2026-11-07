import numpy as np
from scipy.stats import norm
import time

# Track call count and time for safety
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Shekel 4d.
    
    Combines Stochastic Expected Improvement (SEI) with a differential cost
    penalty. The SEI models cost as a log-normal random variable, integrating
    cost uncertainty into the acquisition integral. Additionally, a Gaussian
    penalty term exp(-(beta * cost)^2 / 2) is applied to differentially penalize
    high-cost candidates more strongly than the deterministic EI/cost ratio.
    """
    global _call_count, _start_time
    
    # Initialize timing on first call
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 300:  # Safety margin before 330s limit
        return 1.0 / cost
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for improvement: positive when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Model cost as log-normal random variable
    # cost is in [1, 2], treat as LogNormal with mean=cost and small std
    cost_mean = cost
    cost_std = 0.1 * cost_mean
    
    # Log-normal parameters
    var = cost_std ** 2
    sigma_c_sq = np.log(1.0 + var / (cost_mean ** 2))
    sigma_c = np.sqrt(sigma_c_sq)
    mu_c = np.log(cost_mean) - 0.5 * sigma_c_sq
    
    # E[C^{-1}] for C ~ LogNormal(mu_c, sigma_c) is:
    # exp(-mu_c + 0.5 * sigma_c^2)
    e_inv_c = np.exp(-mu_c + 0.5 * sigma_c_sq)
    
    # SEI = EI * E[C^{-1}]
    sei = ei * e_inv_c
    
    # Differential cost penalty: exp(-(beta * cost)^2 / 2)
    # This penalizes high-cost points more than the linear 1/cost penalty
    # beta = 1.0 gives moderate differential penalty
    beta = 1.0
    cost_penalty = np.exp(-0.5 * (beta * cost) ** 2)
    
    # Combine SEI with differential cost penalty
    score = sei * cost_penalty
    
    # Add a small exploration bonus: prefer high-sigma regions when EI is low
    # This helps in early stages
    sigma_scale = np.abs(f_best) + 1.0
    rel_sigma = sigma_safe / sigma_scale
    
    # Exploration bonus: small, proportional to relative sigma, weighted by cost penalty
    exploration_bonus = 0.05 * rel_sigma * cost_penalty
    
    score = score + exploration_bonus
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
