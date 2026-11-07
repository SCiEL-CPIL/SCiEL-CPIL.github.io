"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Probability of Improvement (PI) instead of EI.
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is less prone to over-exploring than EI when posterior is poorly constrained.
    z = (f_best - mu) / (sigma + 1e-9)
    pi_scores = norm.cdf(z)
    
    # Add a small exploration bonus based on uncertainty to avoid pure exploitation
    # The bonus is scaled by the standard deviation to encourage exploring high-uncertainty regions
    # but damped to not overwhelm the exploitation signal
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma / (np.max(sigma) + 1e-9)
    
    # Combine PI with a mild exploration term
    scores = pi_scores + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
