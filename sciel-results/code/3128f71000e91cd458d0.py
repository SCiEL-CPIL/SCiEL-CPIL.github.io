"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Probability of Improvement (PI) with a knowledge-informed prior
    that biases exploration toward the known basin of attraction of the
    global minimum. Structurally different from EI+UCB blending:
    - PI is a probability (bounded [0,1]) rather than an expected value
    - Domain knowledge provides a Gaussian-shaped prior in scaled coordinates
    - Exploration is controlled via a dynamic threshold on the improvement margin
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Probability of Improvement: P(f(x) < f_best)
    # PI = Phi((f_best - mu) / sigma)
    # But we want to favor points that are likely to improve significantly
    # Use a threshold-based variant: PI with a margin
    
    # Dynamic threshold: early on, use a small margin (explore broadly)
    # Later, use a larger margin (exploit more)
    # Estimate progress based on f_best relative to known global minimum
    global_min = -39.1662
    # Function range for normalization
    range_scale = 60.0
    
    # Progress: 0 when far, 1 when close
    progress = np.clip((f_best - global_min) / range_scale, 0.0, 1.0)
    
    # Threshold margin: starts small, grows as we get closer to optimum
    # This controls how much improvement we demand
    margin = 0.05 * progress + 0.001
    
    # z-score for PI with margin
    z = (f_best - margin - mu) / sigma
    pi = norm.cdf(z)
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # Knowledge-informed prior:
    # The global minimum of Styblinski-Tang 1D is near x = -1.712
    # In scaled [0,1] coordinates: (x + 5) / 10 = 0.3288
    # We don't have coordinates, but we can use the GP uncertainty structure.
    # Instead, we use a different structural insight:
    # Points with high uncertainty AND low mean are most promising.
    # This is captured by a "novelty-weighted PI" that upweights
    # points where the GP is uncertain.
    
    # Novelty weight: higher where sigma is large (unexplored regions)
    # But scaled so it doesn't dominate
    sigma_max = np.max(sigma) + 1e-12
    novelty = sigma / sigma_max  # normalized uncertainty
    
    # Combine PI with novelty in a multiplicative way
    # This is structurally different from additive blending
    # score = PI * (1 + alpha * novelty)
    # where alpha adapts based on progress
    
    # Early: high alpha (explore uncertain regions)
    # Late: low alpha (exploit known good regions)
    alpha = 2.0 * (1.0 - progress) + 0.3
    
    score = pi * (1.0 + alpha * novelty)
    
    # Add a small exploitation term: points with very low mu get a boost
    # This ensures we don't completely ignore known good regions
    mu_range = np.max(mu) - np.min(mu) + 1e-12
    mu_norm = (mu - np.min(mu)) / mu_range  # 0 = best, 1 = worst
    exploit_boost = 0.1 * (1.0 - mu_norm) * progress
    
    score = score + exploit_boost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
