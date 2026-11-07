import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D.
    
    Uses a time-aware UCB (Upper Confidence Bound) acquisition that balances
    exploitation and exploration. The beta parameter adapts based on the
    number of observations (estimated from sigma magnitude) and the gap between
    f_best and the current mean predictions.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Estimate progress from the magnitude of sigma
    # Early in optimization, sigma is large; later it shrinks
    # Use the mean sigma as a proxy for number of evaluations
    mean_sigma = np.mean(sigma)
    
    # Typical sigma range: starts around 1-2 (standardized), drops to ~0.1-0.3
    # Map this to a progress estimate
    # If mean_sigma > 0.5, we're early; if < 0.1, we're late
    progress = np.clip(1.0 - (mean_sigma - 0.1) / 0.9, 0.0, 1.0)
    
    # Adaptive beta: higher early (exploration), lower late (exploitation)
    # Beta ranges from ~3.0 early to ~1.5 late
    beta = 1.5 + 1.5 * progress
    
    # Compute UCB for minimization (we want to maximize the score)
    # UCB = -mu + beta * sigma (higher score = more promising)
    scores = -mu + beta * sigma
    
    # Add a small EI component to help with exploitation
    Z = (f_best - mu) / sigma
    Z = np.clip(Z, -10, 10)
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Combine: primarily UCB, with a small EI boost
    scores = scores + 0.1 * EI
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are reasonably bounded
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
