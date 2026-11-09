"""Bayesian optimization acquisition for a 2D RBF mixture (Matern-5/2 GP)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 325.0          # safety margin before the 330 s kill
_EPS = 1e-9


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _T0) > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Return 2000 finite scores; higher = evaluate next.
    Uses multi-sample Thompson Sampling with min-aggregation to approximate
    the probability of discovering the global minimum basin.
    """
    n = len(mu)
    
    # Number of posterior samples to draw
    # More samples better estimate the probability of being the global minimizer
    n_samples = 8
    
    # Draw multiple posterior samples: f_k(x) ~ N(mu(x), sigma(x)^2)
    # Shape: (n_samples, n)
    Z = np.random.randn(n_samples, n)
    samples = mu[np.newaxis, :] + sigma[np.newaxis, :] * Z
    
    # For each candidate, take the minimum across samples (soft-min approximation)
    # This favors candidates that have at least one very low sample,
    # approximating the expected improvement in the global minimum
    min_samples = np.min(samples, axis=0)
    
    # Also compute a weighted combination:
    # -min_samples gives exploration (probability of global improvement)
    # -mu gives exploitation (low posterior mean)
    # -sigma gives uncertainty bonus
    exploit = -mu
    explore = -min_samples
    
    # Blend: primarily use the min-sample score, with a small exploitation bias
    # The coefficient 0.15 adds mild preference for low-mean points among
    # those with similar min-sample scores
    score = explore + 0.15 * exploit
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
