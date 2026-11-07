"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining Expected Improvement (EI)
    with a UCB-like exploration bonus that decays as f_best improves.
    
    The base score is EI, which drives exploitation toward known good regions.
    An additional exploration term beta * sigma is added, where beta is
    inversely proportional to the current regret (f_best - 0), so early
    in optimization (large f_best) we explore more, and late (small f_best)
    we exploit more. This mirrors the adaptive kappa strategy that showed
    improvement.
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores for each candidate. Higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration bonus:
    # beta is large when f_best is far from 0 (early), small when close (late)
    # Use a smooth decay: beta = base_beta / (1 + f_best)
    # For Ackley 2D, f_best starts around 1-5 and approaches 0
    base_beta = 0.5
    beta = base_beta / (1.0 + max(f_best, 0.0))
    
    # Add exploration bonus scaled by sigma
    exploration_bonus = beta * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
