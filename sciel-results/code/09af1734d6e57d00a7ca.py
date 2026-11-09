"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) acquisition function.
    # Uses a dynamic exploration parameter beta that adapts to the global uncertainty
    # of the Gaussian Process (mean sigma).
    
    # Calculate global uncertainty
    mean_sigma = np.mean(sigma)
    
    # Adaptive beta:
    # When mean_sigma is high (early/exploratory phase), beta is high (max 3.0).
    # When mean_sigma is low (late/exploitative phase), beta is low (min 1.0).
    # The scaling factor 0.5 is chosen based on typical sigma magnitudes for Powell 4D.
    beta = 1.0 + 2.0 * (mean_sigma / (mean_sigma + 0.5))
    
    # UCB Score: -mu + beta * sigma
    # We minimize, so we want low mu. Hence -mu.
    # We want high sigma to explore.
    ucb_score = -mu + beta * sigma
    
    # Cost normalization: divide by cost to prioritize cheaper evaluations.
    # Cost is in [1, 2].
    cost_safe = np.maximum(cost, 1e-6)
    
    scores = ucb_score / cost_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
