"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu)/sigma
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Hypervolume-based: use a reference point that's worse than f_best
    # to encourage exploration of potentially better regions
    # f_ref = f_best + alpha * mean_sigma
    # This makes the "improvement" relative to a worse reference,
    # which gives more weight to high-uncertainty regions
    
    mean_sigma = np.mean(sigma)
    if mean_sigma > 1e-6:
        alpha = 0.5
        f_ref = f_best + alpha * mean_sigma
    else:
        f_ref = f_best
    
    # EHVI with adaptive reference
    z_hv = (f_ref - mu) / sigma
    ehvi = (f_ref - mu) * norm.cdf(z_hv) + sigma * norm.pdf(z_hv)
    
    # Combine EI and EHVI: use a weighted sum
    # EI captures immediate improvement, EHVI encourages exploration
    # Weight them equally to balance exploitation and exploration
    scores = 0.5 * ei + 0.5 * ehvi
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
