"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Pure GP-LCB acquisition for minimization.
    # We want to minimize the function f(x).
    # The Lower Confidence Bound is L(x) = mu(x) - kappa * sigma(x).
    # We select x to minimize L(x).
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the score S(x) = -L(x) = -mu(x) + kappa * sigma(x).
    # This encourages exploration in regions with high uncertainty (high sigma)
    # and exploitation in regions with low predicted value (low mu).
    
    # Fixed exploration parameter
    kappa = 2.0
    
    # Calculate scores
    scores = -mu + kappa * sigma
    
    return scores
# EVOLVE-BLOCK-END
