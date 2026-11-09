"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a posterior-sampling approximation of
    Expected Improvement. Draws S latent function samples from the GP posterior
    at the candidate points and estimates EI as the mean improvement over
    f_best across those samples. Higher scores mean higher expected gain.
    """
    # Deadline guard
    t0 = time.time()
    deadline = 300.0 - 2.0  # safety margin

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    n = len(mu)

    # Check deadline before computation
    if time.time() > deadline:
        return -mu

    # Number of posterior samples for EI approximation
    S = 50

    # Draw S samples: shape (S, n)
    # y_s(i) ~ N(mu_i, sigma_i^2)
    rng = np.random.default_rng()
    # Standard normals of shape (S, n)
    z = rng.standard_normal((S, n))
    # samples[s, i] = mu[i] + sigma[i] * z[s, i]
    samples = mu[None, :] + sigma[None, :] * z  # shape (S, n)

    # Improvement over f_best: imp_s(i) = max(0, f_best - samples[s, i])
    # Since we're minimizing, improvement is when sample < f_best
    improvements = np.maximum(0.0, f_best - samples)  # shape (S, n)

    # EI estimate: mean improvement over S samples
    scores = np.mean(improvements, axis=0)  # shape (n,)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
