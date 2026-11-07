"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    with adaptive exploration. The UCB term helps escape local optima and explore
    under-explored regions, while EI focuses on exploitation. We blend them with
    a weight that decreases over time based on the gap between f_best and the
    current posterior mean/sigma.
    
    mu: (2000,) posterior mean in objective units
    sigma: (2000,) posterior standard deviation in objective units
    f_best: scalar, lowest observed value
    
    Returns: (2000,) scores, higher = more promising
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound for minimization (lower is better, so we want low mu + high uncertainty)
    # UCB for minimization: mu - beta * sigma
    # We use an adaptive beta that starts high and decreases
    # Use a simple heuristic: beta based on the spread of sigma
    beta = 2.0  # Fixed UCB parameter
    ucb = mu - beta * sigma
    
    # Normalize both to comparable scales
    # EI is always >= 0, UCB can be negative
    # We'll use a weighted sum with adaptive weights
    
    # Compute a measure of how "explored" we are: if f_best is much lower than typical mu, we're exploiting
    # If sigma is large relative to (f_best - mu), we should explore more
    
    # Adaptive weight: when EI is small (near optimum), explore more
    # When EI is large (far from optimum), exploit more
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-10
    ei_norm = ei / ei_max
    
    # Weight for UCB increases when EI is small (we're close to best)
    # Use a smooth transition
    weight_ucb = 0.3 + 0.7 * (1.0 - ei_norm)
    weight_ei = 1.0 - weight_ucb
    
    # Combine: we want higher scores to be better
    # EI: higher is better (more improvement expected)
    # UCB: lower is better (lower mu - beta*sigma means more promising for minimization)
    # So we negate UCB to make higher better
    score = weight_ei * ei - weight_ucb * ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
