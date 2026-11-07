"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines Expected Improvement with an
    exploration bonus based on the posterior standard deviation.
    
    This is a variant of Expected Improvement that explicitly rewards
    high-uncertainty regions, which can help escape local optima in
    the Ackley function's complex landscape.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Base Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Exploration bonus: reward high uncertainty regions
    # This encourages sampling in areas where we know less
    # The coefficient balances exploitation (EI) vs exploration (sigma)
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combined score: higher is better
    score = ei + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
