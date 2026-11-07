"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Compute Upper Confidence Bound (UCB) acquisition scores for minimization.
    
    UCB(x) = -(mu(x) - kappa * sigma(x))
    
    This is equivalent to minimizing the lower confidence bound LCB(x) = mu(x) - kappa * sigma(x).
    Higher scores mean more promising candidates to evaluate next.
    
    We use a high kappa value to promote strong exploration, which is necessary
    for escaping local minima in the multimodal Weierstrass function.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid issues with zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # High kappa for strong exploration in multimodal landscape
    kappa = 2.5
    
    # LCB = mu - kappa * sigma
    # We want to MINIMIZE the objective, so we maximize -LCB = -mu + kappa * sigma
    ucb_score = -mu + kappa * sigma_safe
    
    # Ensure finite output
    ucb_score = np.nan_to_num(ucb_score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return ucb_score
# EVOLVE-BLOCK-END
