"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses Posterior Sampling (Thompson Sampling) with differential cost-aware scaling:
    - Exploitation term (f_best - mu) gets a mild cost penalty (division by cost^0.3)
    - Exploration term (sigma) gets an inverse-sqrt cost scaling (division by sqrt(cost))
    
    This treats expensive regions as higher-variance for exploration while still
    exploiting cheap areas, and encourages exploring cheap uncertain regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Guard against division by zero and non-finite values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Ensure inputs are finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Thompson Sampling: sample from posterior
    # Draw random normal samples for exploration
    # Use a fixed seed based on f_best to make it deterministic per call context
    # but vary across calls as f_best changes
    seed_val = int(np.abs(f_best * 1000)) % (2**32)
    rng = np.random.RandomState(seed_val)
    epsilon = rng.normal(0, 1, size=mu.shape)
    
    # Posterior sample: theta = mu + sigma * epsilon
    theta = mu + sigma_safe * epsilon
    
    # Differential cost-aware scaling:
    # Exploitation term: (f_best - theta) / cost^0.3  (mild penalty)
    # Exploration term: sigma / sqrt(cost)  (inverse-sqrt scaling)
    
    # Compute exploitation component
    exploitation = (f_best - theta) / np.power(cost_safe, 0.3)
    
    # Compute exploration component
    exploration = sigma_safe / np.sqrt(cost_safe)
    
    # Adapt the balance based on how close f_best is to the minimum (0)
    # When f_best is large (early), explore more; when small (late), exploit more
    f_best_norm = np.clip(f_best / 20.0, 0.0, 1.0)
    
    # Exploration weight: higher when f_best is large
    # alpha controls the relative weight of exploration vs exploitation
    alpha = 0.5 + 1.5 * f_best_norm  # ranges from 0.5 to 2.0
    
    # Combine: higher score = more desirable
    scores = exploitation + alpha * exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
