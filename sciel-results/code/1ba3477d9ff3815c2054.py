"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Lower bound of Expected Improvement (LBEI) with a small exploration bonus.
    Uses the lower bound of EI which is more robust to noise and avoids
    over-exploration of regions with high uncertainty but low potential.
    """
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-8)
    
    # Standardized distance from the best
    z = (f_best - mu) / sigma
    
    # Lower bound of EI: (f_best - mu) * Phi(z)
    # This is always non-negative when mu < f_best, and 0 otherwise
    # It's a conservative estimate that favors exploitation
    
    lb_ei = (f_best - mu) * norm.cdf(z)
    
    # Add a small exploration term to avoid getting stuck in local optima
    # Use a small fraction of the uncertainty to encourage some exploration
    exploration = 0.01 * sigma
    
    # Combine: primarily exploitation via LBEI, with slight exploration
    scores = lb_ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
