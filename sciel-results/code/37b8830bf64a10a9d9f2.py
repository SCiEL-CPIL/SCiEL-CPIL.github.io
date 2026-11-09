"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330 - 30  # 300s wall-clock limit, 30s safety margin


def acquisition(mu, sigma, f_best):
    # Combined Expected Improvement + small UCB exploration bonus
    # EI balances exploitation and exploration via normal CDF/PDF
    # UCB term helps escape local minima in multimodal Goldstein-Price
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Small UCB component for exploration
    kappa_ucb = 0.1
    ucb = -mu + kappa_ucb * sigma
    
    # Weighted combination: mostly EI with some UCB
    return ei + 0.5 * ucb
# EVOLVE-BLOCK-END
