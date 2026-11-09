# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin: evaluator kills at 330s, we allow 300s per eval
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For points where improvement <= 0, EI should be 0 (or very small)
    ei = np.where(improvement > 0, ei, 0.0)
    
    # Add a small exploration bonus based on uncertainty
    # This helps balance exploitation and exploration
    # Use a weighted combination: primarily EI, with a small UCB-like term
    # The UCB term encourages exploring high-uncertainty regions
    ucb_weight = 0.1  # Small weight for exploration
    ucb_term = sigma_safe  # Uncertainty-based term
    
    # Combine: primarily EI, with slight exploration bonus
    scores = ei + ucb_weight * ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
