"""Acquisition function for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition combining Expected Improvement with a strong
    uncertainty bonus that scales with exploration needs.
    
    We use a dynamic UCB-like term combined with EI to balance
    exploitation and exploration more effectively than pure EI.
    """
    # Ensure sigma is positive and avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound with adaptive kappa
    # Use a larger kappa to encourage exploration in early stages
    # and smaller kappa to focus on exploitation later
    # Since we don't know the iteration number, we use a fixed but 
    # larger kappa than typical UCB values
    
    # Dynamic kappa: start with high exploration, decrease over time
    # We can estimate progress by comparing mu to f_best
    # If mu is close to f_best, we're in exploitation mode
    # If mu is far from f_best, we need more exploration
    
    # Compute a measure of how much improvement is possible
    # Higher sigma relative to (f_best - mu) suggests more uncertainty
    uncertainty_ratio = sigma / (np.abs(f_best - mu) + 1e-8)
    
    # Use a combination of EI and UCB
    # UCB with kappa that adapts based on uncertainty
    kappa = 1.5  # Strong exploration bonus
    
    # UCB score (we want to minimize, so higher UCB means more uncertain/potential)
    ucb = mu - kappa * sigma
    
    # Combine EI and UCB: use EI for exploitation, UCB for exploration
    # The idea: when EI is high (good exploitation opportunity), weight EI more
    # When EI is low (no clear improvement), weight UCB more (explore)
    
    # Normalize EI to [0, 1] range approximately
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_normalized = ei / (ei_max + 1e-8)
    
    # Weight: more EI when exploitation is promising, more UCB otherwise
    # Use a sigmoid-like weighting based on EI quality
    weight = 0.7 * ei_normalized + 0.3 * (1 - ei_normalized)
    
    # Final score: weighted combination
    # We want higher score = better candidate
    # EI is already higher for better candidates
    # For UCB, lower mu - kappa*sigma is better (more negative = more promising)
    # So we negate UCB to make higher = better
    score = weight * ei - (1 - weight) * (mu - kappa * sigma)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
