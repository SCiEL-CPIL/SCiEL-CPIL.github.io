"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = time.time() + 320  # Safety margin before 330s wall-clock limit


def acquisition(mu, sigma, f_best):
    """
    Posterior-sampling-based acquisition function that approximates Expected Improvement.
    
    Draws 50 latent function samples from the GP posterior and takes the maximum
    improvement over f_best across these samples as the acquisition score.
    """
    n = len(mu)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw posterior samples: f(x) ~ N(mu(x), sigma^2(x))
    # We draw 50 samples and compute max improvement
    n_samples = 50
    
    # Standardize for numerical stability
    # f_best is the current best (lowest) value
    # Improvement at x is max(0, f_best - f(x))
    # E[improvement] ≈ (1/n_samples) * sum_s max(0, f_best - f_s(x))
    # But the task says "taking the maximum improvement over f_best across these samples"
    # This means for each candidate x, we compute max over samples of (f_best - f_s(x))
    # and then take max with 0? Or just max(f_best - f_s(x)) across samples?
    
    # Let me re-read: "taking the maximum improvement over f_best across these samples"
    # I interpret this as: for each candidate x, score(x) = max_s (f_best - f_s(x))
    # This is a stochastic upper bound on improvement
    
    # Generate random numbers for all samples at once for efficiency
    # Shape: (n_samples, n)
    eps = np.random.randn(n_samples, n)
    
    # f_s(x) = mu(x) + sigma(x) * eps_s
    # Shape: (n_samples, n)
    f_samples = mu[np.newaxis, :] + sigma_safe[np.newaxis, :] * eps
    
    # Improvement for each sample: f_best - f_s(x)
    # Shape: (n_samples, n)
    improvements = f_best - f_samples
    
    # Maximum improvement across samples for each candidate
    # Shape: (n,)
    scores = np.max(improvements, axis=0)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
