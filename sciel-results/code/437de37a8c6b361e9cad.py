# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track call count for adaptive exploration
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses an adaptive strategy that balances exploration and exploitation.
    Early on, uses UCB with high exploration; later shifts to EI-like exploitation.
    """
    global _call_count
    _call_count += 1
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: starts high for exploration, decays for exploitation
    # With ~25 acquisition calls (30 evals - 5 initial), decay from 4.0 to 1.0
    n_calls = _call_count
    # Decay beta from 4.0 to 1.0 over 20 calls
    if n_calls <= 20:
        beta = 4.0 - (3.0 * (n_calls - 1) / 19.0)
    else:
        beta = 1.0
    
    # UCB acquisition: mu + beta * sigma
    ucb = mu + beta * sigma
    
    # Also compute EI for comparison
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Blend: early on prefer UCB, later prefer EI
    # Use a smooth transition
    if n_calls <= 5:
        # Pure UCB for first few calls
        scores = ucb
    elif n_calls <= 15:
        # Mix UCB and EI
        # Normalize both to [0, 1] range for mixing
        ucb_min, ucb_max = np.min(ucb), np.max(ucb)
        ei_min, ei_max = np.min(ei), np.max(ei)
        
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min + 1e-9)
        ei_norm = (ei - ei_min) / (ei_max - ei_min + 1e-9)
        
        # Weight shifts from UCB to EI
        w_ucb = max(0.0, 1.0 - (n_calls - 5) / 10.0)
        w_ei = 1.0 - w_ucb
        
        scores = w_ucb * ucb_norm + w_ei * ei_norm
    else:
        # Pure EI for later calls
        scores = ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
