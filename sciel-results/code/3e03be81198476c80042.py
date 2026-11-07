"""Rewritten with Thompson Sampling acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    n = len(mu)
    
    # Deadline guard: aim to finish well before 300s
    start_time = time.time()
    deadline = 280.0  # safety margin
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample K latent function values from GP posterior
    # f(x) ~ N(mu(x), sigma(x)^2) at each candidate
    # We sample K times and take the negative minimum as score
    
    # Number of samples - balance between quality and speed
    K = 50  # reduced from 100 for speed; still provides good exploration
    
    # For each candidate i, sample from N(mu[i], sigma[i]^2)
    # We can vectorize: generate K x n random samples
    # But K*n = 50*2000 = 100000, which is fine
    
    rng = np.random.RandomState(42)  # fixed seed for reproducibility within a call
    
    # Generate standard normal samples: shape (K, n)
    Z = rng.standard_normal((K, n))
    
    # Sampled values: shape (K, n)
    samples = mu[None, :] + sigma_safe[None, :] * Z
    
    # For each candidate, take the minimum over K samples
    # Lower sampled value = more promising for minimization
    min_samples = samples.min(axis=0)  # shape (n,)
    
    # Score: higher means evaluate next, so negate the minimum
    scores = -min_samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
