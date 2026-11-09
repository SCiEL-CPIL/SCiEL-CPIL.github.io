"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()
# Track number of acquisition calls to adapt exploration
_call_count = 0


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Posterior-sampling (Thompson sampling) acquisition for Bayesian optimization.
    
    Structural change from deterministic EI: instead of computing expected
    improvement, we draw multiple samples from the GP posterior and score
    each candidate by how often it is the minimum across samples, blended
    with an exploitation score based on mu and sigma.
    """
    global _call_count
    _call_count += 1

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Check deadline: if we're running out of time, return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        scores = -mu + 0.05 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Adaptive exploration parameter: start more exploratory, become exploitative
    # Total evaluations = 30, first 5 are initial random points
    # So we have ~25 acquisition calls
    progress = min(_call_count / 25.0, 1.0)
    
    # Number of posterior samples: use more in early stages
    # Keep it modest for speed (each sample is O(n))
    n_samples = max(8, int(20 * (1.0 - 0.5 * progress)))
    
    # Cap total work: n_samples * n should be reasonable
    # n=2000, so n_samples up to 20 gives 40000 ops per call - fine
    
    # Generate random normal samples
    rng = np.random.default_rng()
    
    # Posterior sampling: for each sample, draw y_i ~ N(mu_i, sigma_i^2)
    # Shape: (n_samples, n)
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, n))
    
    # For each sample, find the index of the minimum
    # min_indices shape: (n_samples,)
    min_indices = np.argmin(samples, axis=1)
    
    # Count how often each candidate is the minimum
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(counts, min_indices, 1.0)
    
    # Normalize to [0, 1]
    max_count = counts.max()
    if max_count > 0:
        counts = counts / max_count
    
    # Exploitation score: lower mu is better, higher sigma adds uncertainty bonus
    # Scale mu relative to f_best for better conditioning
    mu_range = f_best - mu.min()
    if mu_range > 1e-12:
        mu_norm = (f_best - mu) / mu_range
    else:
        mu_norm = np.zeros_like(mu)
    
    # Sigma bonus: relative to typical sigma
    sigma_scale = np.median(sigma)
    if sigma_scale > 1e-12:
        sigma_norm = sigma / sigma_scale
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Blend: exploration (posterior sampling frequency) + exploitation (mu, sigma)
    # As progress increases, weight exploitation more
    w_exploit = 0.3 + 0.5 * progress  # 0.3 to 0.8
    w_explore = 1.0 - w_exploit       # 0.7 to 0.2
    
    scores = w_exploit * (mu_norm + 0.1 * sigma_norm) + w_explore * counts
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
