"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Stochastic acquisition function combining Expected Improvement (EI) with
    Thompson Sampling-like noise injection for better exploration.
    
    Uses a hybrid approach: compute EI scores but add a small stochastic component
    to encourage exploration in high-uncertainty regions.
    """
    # Ensure sigma is non-negative and handle any numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    from scipy.stats import norm
    
    # Compute the improvement potential
    improvement = f_best - mu
    
    # Standardize
    z = improvement / sigma
    
    # Compute EI
    # Handle cases where z is very large or very small to avoid numerical issues
    ei = np.zeros_like(mu)
    
    # For points where improvement > 0 (i.e., mu < f_best)
    mask = z > -10  # Only compute where there's reasonable chance of improvement
    z_clipped = np.clip(z[mask], -10, 10)
    
    # EI formula: improvement * Phi(z) + sigma * phi(z)
    ei[mask] = improvement[mask] * norm.cdf(z_clipped) + sigma[mask] * norm.pdf(z_clipped)
    
    # For points with very negative z (mu >> f_best), EI is essentially 0
    
    # Add a stochastic exploration component based on uncertainty
    # This encourages exploration in high-uncertainty regions
    rng = np.random.default_rng()
    exploration_weight = 0.1  # Weight for exploration vs exploitation
    
    # Draw random samples to add exploration
    exploration = rng.standard_normal(len(mu)) * sigma * exploration_weight
    
    # Combine EI (exploitation) with exploration noise
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
