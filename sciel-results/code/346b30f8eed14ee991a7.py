import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Thompson Sampling proxy via normalized improvement heuristic
    # Score = (f_best - mu) / (sigma + epsilon)
    # This ratio is high when the expected improvement is large relative to uncertainty,
    # encouraging exploration of regions with high potential but not overly penalized by noise.
    # We add a small constant to avoid division by zero and ensure numerical stability.
    epsilon = 1e-9
    scores = (f_best - mu) / (sigma + epsilon)
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
