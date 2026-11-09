"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = time.time() + 320  # Safety margin before 330s wall-clock limit
_START = time.time()
_call_count = 0


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Posterior-sampling-based acquisition function that approximates Expected Improvement.
    
    Draws 50 latent function samples from the GP posterior and computes the mean
    improvement over f_best across these samples as the acquisition score.
    Blends with a small UCB term for balanced exploration/exploitation.
    """
    global _call_count
    _call_count += 1
    
    n = len(mu)
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline: if we're running out of time, return a simple exploit score
    if _elapsed() > 315.0:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Number of posterior samples
    n_samples = 50
    
    # Generate random numbers for all samples at once for efficiency
    # Shape: (n_samples, n)
    rng = np.random.default_rng()
    eps = rng.standard_normal((n_samples, n))
    
    # f_s(x) = mu(x) + sigma(x) * eps_s
    # Shape: (n_samples, n)
    f_samples = mu[np.newaxis, :] + sigma[np.newaxis, :] * eps
    
    # Improvement for each sample: f_best - f_s(x), clipped at 0
    # Shape: (n_samples, n)
    improvements = np.maximum(f_best - f_samples, 0.0)
    
    # Mean improvement across samples = estimate of EI
    # Shape: (n,)
    ei_scores = np.mean(improvements, axis=0)
    
    # Add a small UCB component for exploration balance
    # Scale the UCB term relative to the EI scale to avoid dominating
    ei_scale = np.max(ei_scores) if np.max(ei_scores) > 1e-12 else 1.0
    ucb_term = 0.1 * sigma / max(ei_scale, 1e-9) * ei_scale
    
    # Blend: primarily EI, with a small UCB boost for uncertain regions
    scores = ei_scores + ucb_term
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
