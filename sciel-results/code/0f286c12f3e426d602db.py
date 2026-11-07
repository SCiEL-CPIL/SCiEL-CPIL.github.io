"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an exploration bonus
    that adapts to the current uncertainty level. This helps balance exploitation
    (finding better optima) with exploration (visiting uncertain regions).
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure no negative sigma or NaN/Inf
    sigma = np.maximum(sigma, 1e-9)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration bonus:
    # - When sigma is large relative to the spread of f_best - mu, encourage exploration
    # - The beta parameter adapts based on the ratio of max sigma to typical improvement
    #   This helps escape local minima by weighting uncertain regions more heavily early on
    
    # Compute a scale for adaptive beta
    sigma_max = np.max(sigma)
    sigma_mean = np.mean(sigma)
    
    # Beta that encourages exploration when uncertainty is high
    # Larger beta -> more exploration; smaller beta -> more exploitation
    # We use a smooth function that increases with relative uncertainty
    uncertainty_ratio = sigma_max / (sigma_mean + 1e-9)
    
    # Base beta around 0.5, scaling up with uncertainty ratio
    # This provides a smooth transition from exploitation to exploration
    beta = 0.3 + 0.5 * np.tanh(uncertainty_ratio / 2.0)
    
    # Lower Confidence Bound component (exploration)
    lcb = mu + beta * sigma
    
    # Combine: use a weighted sum where EI dominates when improvement is clear
    # and LCB dominates when uncertainty is high
    # The weight shifts based on how much improvement EI suggests vs uncertainty
    
    # Normalize for combination
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-9
    lcb_max = np.max(lcb) if np.max(lcb) > 0 else 1e-9
    
    # Weight: when EI is large relative to LCB, trust EI more
    # When LCB dominates (high uncertainty), trust exploration more
    ei_weight = 0.7  # Base weight for exploitation
    lcb_weight = 0.3  # Base weight for exploration
    
    # Adjust weights based on current state
    # If the best EI is small, increase exploration weight
    if ei_max < 1e-6:
        ei_weight = 0.3
        lcb_weight = 0.7
    elif ei_max < 0.01:
        ei_weight = 0.5
        lcb_weight = 0.5
    
    # Final score: normalized combination
    scores = ei_weight * (ei / (ei_max + 1e-9)) + lcb_weight * (lcb / (lcb_max + 1e-9))
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
