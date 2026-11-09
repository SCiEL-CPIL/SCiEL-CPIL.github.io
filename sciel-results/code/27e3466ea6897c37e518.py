"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()

# RNG for reproducible posterior sampling within a run
# Using a fixed seed ensures reproducibility across calls if the evaluator re-imports,
# but typically we want fresh randomness per acquisition call to avoid correlation.
# However, creating a new RNG every call has overhead. 
# A global RNG is standard. We will use a global one but ensure it's not exhausted in a bad way.
_rng = np.random.default_rng(42)


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a posterior-sampling based 'Max-Improvement' strategy.
    
    This approach draws multiple samples from the GP posterior. For each candidate,
    it computes the improvement over f_best for each sample and takes the maximum.
    This acts as an optimistic/UCB-like acquisition function, encouraging exploration
    in regions with high uncertainty that could potentially yield the global minimum.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Check deadline: if we're running out of time, return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: exploit (lowest mu) with tiny exploration
        scores = -mu + 0.1 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Number of posterior samples
    # 50 samples is a standard trade-off between computational cost and approximation quality.
    n_samples = 50
    n_candidates = mu.size

    # Draw posterior samples: Y_s ~ N(mu, sigma^2) for s in 1..n_samples
    # Shape: (n_samples, n_candidates)
    # Using standard_normal for better performance and less overhead than normal
    eps = _rng.standard_normal((n_samples, n_candidates))
    samples = mu[None, :] + sigma[None, :] * eps

    # For each sample, compute improvement over f_best
    # improvement_s = max(f_best - Y_s, 0)
    improvements = np.maximum(f_best - samples, 0.0)

    # Structural Change: Use MAX instead of MEAN
    # Taking the maximum across samples creates an optimistic acquisition score.
    # This is equivalent to a stochastic UCB and helps escape local minima.
    scores = np.max(improvements, axis=0)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
