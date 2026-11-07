"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Check time limit to ensure we return within constraints
    if time.time() - _start_time > 320:
        # Fallback: pure exploitation
        return -mu

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = CDF, phi(z) = PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI calculation for minimization
    ei = np.zeros_like(mu)
    mask = mu < f_best
    ei[mask] = (f_best - mu[mask]) * Phi_z[mask] + sigma_safe[mask] * phi_z[mask]
    
    # UCB component for exploration
    # A smaller kappa than 1.0 is used to balance exploitation and exploration
    # since EI already provides some exploration through the sigma term
    kappa = 0.5
    ucb = -mu + kappa * sigma_safe
    
    # Combine EI and UCB: use EI for exploitation and UCB for exploration
    # The combination should favor points with high EI (likely improvement)
    # but also allow some exploration of uncertain regions
    # We normalize both to [0, 1] to make them comparable
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-10:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Weighted combination: favor EI more heavily for exploitation
    # but include UCB to avoid getting stuck in local optima
    weights_ei = 0.7
    weights_ucb = 0.3
    scores = weights_ei * ei_norm + weights_ucb * ucb_norm
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
