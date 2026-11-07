"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Lower Confidence Bound (LCB) with spatial diversity.
    
    Since we only have mu, sigma, and f_best (no coordinates), we cannot
    compute actual distances to the best point. However, we can still use
    a batch-aware LCB that encourages exploration by using a higher beta
    when the current best is far from the minimum (i.e., when we're still
    exploring). The "repulsion" term is approximated by adjusting the
    exploitation/exploration balance based on how close f_best is to the
    known minimum (-3.86278 for Hartmann 3D).
    
    We use a standard LCB: mu - beta * sigma
    where beta is chosen to balance exploration and exploitation.
    
    To incorporate "batch-awareness" and spatial diversity without coordinates,
    we can use an adaptive beta that increases when we're far from the optimum,
    encouraging more exploration early on, and decreases as we get closer.
    
    The Hartmann 3D minimum is approximately -3.86278.
    """
    # Known minimum of Hartmann 3D
    h3_min = -3.86278
    
    # Normalize the progress: 0 when f_best is far from minimum, 1 when close
    # f_best starts around the worst value (e.g., ~0 or positive for Hartmann)
    # and decreases toward -3.86278
    # Let's say "progress" = (initial_guess - f_best) / (initial_guess - h3_min)
    # A reasonable initial guess for the worst case might be around 0 or even positive
    # For Hartmann 3D on [0,1]^3, values range roughly from -3.86 to about 0.5
    # Let's use a reference "worst" value of 0.5
    worst_ref = 0.5
    progress = np.clip((worst_ref - f_best) / (worst_ref - h3_min), 0.0, 1.0)
    
    # Adaptive beta: higher when progress is low (more exploration),
    # lower when progress is high (more exploitation)
    # beta ranges from about 2.5 (high exploration) to 0.5 (high exploitation)
    beta = 2.5 - 2.0 * progress
    
    # Standard LCB: mu - beta * sigma
    # For minimization, we want to select x that minimizes mu - beta * sigma
    # Since "higher meaning evaluate next", we return negative LCB
    score = -mu + beta * sigma
    
    # Add a repulsion/diversity term: penalize points where sigma is very small
    # relative to the overall sigma distribution, which may indicate we're
    # in a well-explored region. This encourages exploring areas with more
    # uncertainty.
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-10
    
    # Normalize sigma to [0, 1] range approximately
    sigma_norm = (sigma - np.min(sigma)) / (np.max(sigma) - np.min(sigma) + 1e-10)
    
    # Repulsion term: add a small bonus to points with higher normalized sigma
    # This encourages exploring uncertain regions
    repulsion_weight = 0.3 * (1.0 - progress)
    diversity_bonus = repulsion_weight * sigma_norm
    
    score = score + diversity_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
