"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Knowledge-gradient-inspired acquisition with adaptive exploration.
    
    The core idea: score each candidate by the expected reduction in f_best
    (i.e., probability of finding a new minimum times expected improvement),
    plus an exploration bonus that is larger in regions with high uncertainty
    relative to the current best.
    """
    t0 = time.time()
    
    # Ensure numerical safety
    sigma = np.maximum(sigma, 1e-9)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Expected Improvement (EI) - how much better this point could be
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of beating f_best
    prob_improve = norm.cdf((f_best - mu) / sigma)
    
    # Knowledge Gradient: expected improvement weighted by probability of improvement
    # This emphasizes points where we're likely to find a new best
    kg = prob_improve * ei
    
    # Adaptive exploration: use the uncertainty structure to guide exploration
    # Scale sigma relative to the spread of mu to get a normalized uncertainty
    mu_spread = np.std(mu) + 1e-9
    sigma_norm = sigma / (mu_spread + 1e-9)
    
    # Exploration bonus: larger where uncertainty is high relative to the landscape
    # Use a softplus-like transformation to keep it smooth
    exploration = np.log1p(sigma_norm) * (1.0 + 0.5 * np.exp(-sigma_norm))
    
    # Combine: primary signal is KG, secondary is exploration
    # The exploration term helps escape local minima by favoring uncertain regions
    scores = 1.5 * kg + 0.8 * exploration
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Time guard: if we're running long, return what we have (shouldn't happen for this simple computation)
    elapsed = time.time() - t0
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
