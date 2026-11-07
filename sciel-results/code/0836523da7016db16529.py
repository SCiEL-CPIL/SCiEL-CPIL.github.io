"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a combination of Expected Improvement (EI)
    and a lower confidence bound (LCB) term, with adaptive exploration.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: lowest observed value so far
    
    Returns: (2000,) array of acquisition scores (higher = more promising)
    """
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) - encourages exploration
    # Use a beta parameter that balances exploitation and exploration
    beta = 2.0
    lcb = mu - beta * sigma
    
    # For minimization, we want low mu and high sigma
    # EI handles this well, but LCB adds extra exploration pressure
    
    # Combine: use EI as primary, but boost points with high uncertainty
    # if they haven't been explored enough
    # Normalize EI and LCB to comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    lcb_min = np.min(lcb)
    lcb_range = np.max(lcb) - lcb_min
    if lcb_range > 1e-12:
        lcb_norm = (lcb_min - lcb) / lcb_range  # Higher for lower LCB values
    else:
        lcb_norm = np.zeros_like(lcb)
    
    # Weighted combination: primarily EI with some LCB influence
    # The LCB term helps escape local optima by favoring uncertain regions
    alpha = 0.7
    score = alpha * ei_norm + (1 - alpha) * lcb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
