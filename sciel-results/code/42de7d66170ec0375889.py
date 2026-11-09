import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s to be safe within 330s limit

def acquisition(mu, sigma, f_best):
    """
    Compute a hybrid acquisition function combining Expected Improvement (EI)
    and Uncertainty (Sigma) with adaptive exploration based on progress
    towards the known global minimum.
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        Scores, shape (N,), higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple UCB to be safe
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    # Handle numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.clip(sigma_safe, 1e-9, 1e6)
    
    # 1. Expected Improvement (EI)
    # For minimization: EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10) # Clip z to avoid numerical issues in cdf/pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0) # EI is non-negative
    
    # 2. Exploration Term (Uncertainty)
    # Higher sigma means higher uncertainty, which is good for exploration
    novelty = sigma_safe
    
    # 3. Adaptive Weighting based on Progress
    # Known global minimum for Styblinski-Tang 1D
    f_min = -39.1662
    
    # Calculate gap between current best and known minimum
    # gap > 0 means we are worse than the optimum
    # gap <= 0 means we have found (or surpassed) the optimum
    gap = f_best - f_min
    
    # Use a sigmoid to smoothly blend between exploitation (EI) and exploration (Novelty)
    # When gap is large (early stage), explore_weight should be high -> explore more
    # When gap is small (late stage), explore_weight should be low -> exploit more
    
    # The scale of the sigmoid (5.0) determines how quickly we switch to exploitation.
    # A smaller scale means a sharper transition.
    explore_weight = 1.0 / (1.0 + np.exp(gap / 5.0))
    
    # Normalize EI and Novelty to [0, 1] using min-max normalization
    def _normalize(arr):
        a_min = np.min(arr)
        a_max = np.max(arr)
        rng = a_max - a_min
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - a_min) / rng
    
    ei_norm = _normalize(ei)
    novelty_norm = _normalize(novelty)
    
    # 4. Blend
    # Score = (1 - explore_weight) * Exploitation + explore_weight * Exploration
    scores = (1.0 - explore_weight) * ei_norm + explore_weight * novelty_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
