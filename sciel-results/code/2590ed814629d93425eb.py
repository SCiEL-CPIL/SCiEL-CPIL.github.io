# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Thompson Sampling with Monte Carlo samples from the GP posterior.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 270  # Safety margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Number of Monte Carlo samples for Thompson Sampling
    # Use a moderate number to balance accuracy and speed
    n_samples = 20
    
    # Generate random normal samples
    rng = np.random.default_rng(42)
    # Shape: (n_samples, 2000)
    samples = rng.standard_normal((n_samples, len(mu)))
    
    # Compute Thompson Sampling samples: y_i = mu + sigma * z_i
    # Shape: (n_samples, 2000)
    ts_samples = mu[None, :] + sigma_safe[None, :] * samples
    
    # For each sample, compute the cost-normalized objective
    # We want to minimize y / cost, so for acquisition (higher is better),
    # we want to maximize the probability of being the best (lowest y/cost)
    
    # Compute cost-normalized values: shape (n_samples, 2000)
    normalized = ts_samples / cost_safe[None, :]
    
    # For each candidate, compute the fraction of samples where it's the minimum
    # This gives the probability of being the best candidate
    # Shape: (2000,)
    is_min = normalized == np.min(normalized, axis=0, keepdims=True)
    scores = np.mean(is_min, axis=0)
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Add a small exploitation component based on expected improvement
    # to help guide search toward promising regions
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1] range
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Blend Thompson Sampling scores with EI
    # Weight more heavily on Thompson Sampling for exploration
    scores = 0.8 * scores + 0.2 * ei_norm / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
