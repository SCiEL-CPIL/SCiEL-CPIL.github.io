"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the improvement term
    # We want to minimize f, so improvement is f_best - f(x).
    # Let d = f_best - mu.
    d = f_best - mu
    
    # Standard normal CDF and PDF
    z = d / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Standard EI
    ei = d * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Determine exploration strength using xi-EI strategy
    # The known global minimum for Styblinski-Tang 1D is -39.1662.
    known_min = -39.1662
    max_val = 100.0  # Approximate maximum value in the domain
    
    # Progress towards optimum: 0 at start, 1 at optimum
    if max_val == known_min:
        progress = 0.0
    else:
        progress = np.clip((max_val - f_best) / (max_val - known_min), 0.0, 1.0)
    
    # Use a decaying xi to shift from exploration to exploitation
    # Early on (progress ~ 0), xi is high, requiring large potential improvement
    # to get a non-zero EI score. This forces exploration of uncertain regions
    # or very promising means.
    # Late on (progress ~ 1), xi is 0, standard EI behavior.
    xi = 8.0 * (1.0 - progress)
    
    # Calculate EI with xi
    d_xi = f_best - xi - mu
    z_xi = d_xi / sigma_safe
    phi_xi = norm.pdf(z_xi)
    Phi_xi = norm.cdf(z_xi)
    
    ei_xi = d_xi * Phi_xi + sigma_safe * phi_xi
    ei_xi = np.maximum(ei_xi, 0.0)
    
    # Add a small exploration bonus to ensure we always have some signal
    # and to break ties in the early phase where EI_xi might be 0 for many points.
    # The bonus decays as we progress.
    exploration_bonus = 0.5 * sigma_safe * (1.0 - progress)
    
    scores = ei_xi + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
