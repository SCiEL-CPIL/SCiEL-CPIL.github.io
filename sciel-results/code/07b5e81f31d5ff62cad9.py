"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) with a small uncertainty boost.
    
    For minimization, PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    We add a small constant times sigma to prevent premature convergence
    in multimodal regions where PI becomes zero.
    """
    # Compute PI score
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    pi_scores = norm.cdf(z)
    
    # Add a small uncertainty boost to encourage exploration
    # This prevents PI from being exactly zero in regions with high uncertainty
    # but low probability of improvement
    boost = 0.05 * sigma / (np.max(sigma) + 1e-10)
    
    # Combine: primarily PI, with a small uncertainty component
    scores = pi_scores + boost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
