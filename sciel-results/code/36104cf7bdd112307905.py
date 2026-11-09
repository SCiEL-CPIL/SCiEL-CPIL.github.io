import numpy as np
import time
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1.0)  # cost is in [1, 2]
    
    # Check deadline
    if time.time() > deadline:
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -50, 50)
        phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
        cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
        ei = (f_best - mu) * cdf_z + sigma * phi_z
        ei = np.maximum(ei, 0.0)
        scores = ei / np.log(1.0 + cost_safe)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    # Compute phi(z) and Phi(z)
    phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    # Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        scores = ei / np.log(1.0 + cost_safe)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Cost-aware Expected Improvement (CEI)
    # Use log(1 + cost) as the penalty - this is a standard approach
    # that smoothly penalizes higher cost without extreme sensitivity
    cost_penalty = np.log(1.0 + cost_safe)
    
    # Base CEI score
    scores = ei / cost_penalty
    
    # Add a small UCB term for exploration, scaled by uncertainty
    # UCB = -mu + kappa * sigma, where kappa adapts to convergence
    sigma_mean = np.mean(sigma)
    mu_std = np.std(mu)
    
    # kappa: higher when more uncertainty relative to model spread
    if mu_std > 1e-9:
        kappa = 1.0 + 2.0 * np.clip(sigma_mean / mu_std, 0, 1)
    else:
        kappa = 2.0
    
    ucb = -mu + kappa * sigma
    ucb = np.maximum(ucb, 0)
    
    # Scale UCB to be comparable to EI
    ei_max = np.max(ei)
    ucb_max = np.max(ucb)
    
    if ucb_max > 1e-9 and ei_max > 1e-9:
        ucb_scale = 0.1 * ei_max / ucb_max  # Small weight for UCB
    else:
        ucb_scale = 0.0
    
    scores += ucb * ucb_scale / cost_penalty
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
