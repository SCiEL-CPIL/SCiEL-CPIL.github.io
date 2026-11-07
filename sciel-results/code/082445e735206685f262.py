"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) with adaptive exploration parameter.
    # This encourages exploration early and exploitation later.
    # Since we don't know the iteration count, we use a fixed but moderate
    # exploration weight that balances exploration and exploitation well
    # for 30 total evaluations.
    
    # Clip sigma to avoid division by zero and ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB: mu + kappa * sigma
    # For minimization, we want to find low values, so we maximize this score
    # A good kappa for ~25 optimization steps (after 5 initial) is around 1.0-1.5
    # Let's use 1.0 which is a standard choice
    kappa = 1.0
    
    ucb = mu + kappa * sigma_safe
    
    # Return finite scores, higher meaning evaluate next
    # Ensure all values are finite
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    return ucb
# EVOLVE-BLOCK-END
