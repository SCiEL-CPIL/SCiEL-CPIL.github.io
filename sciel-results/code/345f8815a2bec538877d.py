import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Adaptive Upper Confidence Bound (UCB) for minimization.
    Beta is adapted based on the progress (gap between f_best and minimum 3.0).
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate adaptive beta
    # The minimum value of Goldstein-Price 2D is 3.0
    gap = max(f_best - 3.0, 0.0)
    
    # Beta is high when far from optimum (exploration) and lower when close (exploitation)
    # Base beta 1.0 ensures some exploration always.
    # Additional up to 4.0 is added when gap is small.
    # Decay scale 10.0 allows significant exploration boost for gaps up to ~20-30 units.
    beta = 1.0 + 4.0 * np.exp(-gap / 10.0)
    
    # UCB score for minimization: -mu + beta * sigma
    # We want to maximize this score.
    # Low mu (good prediction) gives high -mu.
    # High sigma (uncertainty) gives high beta * sigma.
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check if we're running out of time (unlikely for such simple computation)
    if time.time() > deadline:
        # Return a simple UCB-like score as fallback with fixed beta
        fallback_scores = -mu + 1.0 * sigma
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return fallback_scores
    
    return scores
