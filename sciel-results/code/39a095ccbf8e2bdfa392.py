"""Bayesian optimization acquisition for 1d sphere; EVOLVE-BLOCK is the editable part."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def acquisition(mu, sigma, f_best):
    """Knowledge Gradient (KG) approximation for Bayesian optimization of 1D sphere.

    KG estimates the expected reduction in f_best from evaluating at each candidate.
    This is a structural departure from UCB/EI: it directly optimizes the decision
    (which point to evaluate next) rather than using a heuristic score.

    For each candidate x_i with posterior N(mu_i, sigma_i^2), KG_i approximates:
        E[max(f_best, min_{j != i} Y_j) - max(f_best, min(Y_i, min_{j != i} Y_j))]
    where Y_j ~ N(mu_j, sigma_j^2) are independent draws.

    We use a Monte Carlo approximation with a small number of samples for speed.
    """
    t0 = time.perf_counter()
    DEADLINE = 300.0 - 10.0  # safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.size

    # Guard against non-finite inputs.
    finite_mu = np.isfinite(mu)
    finite_sigma = np.isfinite(sigma)
    if not np.all(finite_mu):
        mu = np.where(finite_mu, mu, np.max(mu[finite_mu]) + 1.0 if np.any(finite_mu) else 0.0)
    if not np.all(finite_sigma):
        sigma = np.where(finite_sigma, sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-12)

    # Number of MC samples for KG approximation.
    # Keep small for speed: 50 samples is a good tradeoff.
    n_samples = 50

    # Pre-compute: for each candidate i, we need E[min_{j != i} Y_j] and
    # E[max(f_best, min_{j != i} Y_j)].
    # 
    # Efficient approach: 
    # 1. Draw n_samples independent posterior samples Y_s of shape (n_samples, n).
    # 2. For each sample s, compute m_s = min(Y_s) and its argmin.
    # 3. For each candidate i, KG_i ≈ (1/n_samples) * sum_s [max(f_best, min_{j != i} Y_s) - max(f_best, min(Y_s))]
    #    But this is expensive. Instead, use the fact that for the argmin candidate,
    #    the term is 0, and for others, it's the second-order effect.
    #
    # Simpler and faster: use the analytic approximation for 1D.
    # For each i, the KG contribution is approximately:
    #   sigma_i * phi((f_best - mu_i)/sigma_i) * [1 - Phi((f_best - mu_i)/sigma_i)]
    #   - sigma_i * Phi((f_best - mu_i)/sigma_i) * (1 - Phi((f_best - mu_i)/sigma_i))
    # This is the probability-weighted expected improvement in the minimum.
    #
    # Actually, let's use a cleaner formulation. The KG for candidate i is:
    #   E[max(f_best, min_{j != i} Y_j) - max(f_best, min(Y_i, min_{j != i} Y_j))]
    # ≈ E[max(f_best, M_{-i}) - max(f_best, min(Y_i, M_{-i}))]
    # where M_{-i} = min_{j != i} Y_j.
    #
    # If we approximate M_{-i} by its median or mean, this becomes tractable.
    #
    # Alternative structural change: use **Thompson Sampling** scores.
    # Draw K posterior samples, and for each candidate, compute the fraction of
    # samples in which it achieves the minimum. This is a natural "probability
    # of being the best" score.
    
    K = 100  # number of posterior samples
    rng = np.random.default_rng(42)  # fixed seed for determinism
    
    # Draw K samples from the posterior: shape (K, n)
    Y = mu[np.newaxis, :] + sigma[np.newaxis, :] * rng.standard_normal((K, n))
    
    # For each sample, find the minimum value and its index
    min_vals = np.min(Y, axis=1)  # shape (K,)
    
    # For each candidate i, compute the fraction of samples where i is (among) the minimum
    # This is the "probability of being optimal" under Thompson sampling.
    # To avoid ties, use a small perturbation or exact comparison.
    
    # Count how many times each candidate achieves the minimum
    min_indices = np.argmin(Y, axis=1)  # shape (K,)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(counts, min_indices, 1.0)
    prob_optimal = counts / K  # shape (n,)
    
    # The score is the probability of being optimal. Higher is better.
    # But this alone may be too exploratory. Blend with exploitation.
    
    # Also compute the expected value of the minimum if we pick i:
    # E[min(Y_i, M_{-i})] ≈ use the distribution of Y_i and M_{-i}
    # For simplicity, use: score = prob_optimal * (1 + f_best / (1 + mu_i))
    # No, let's keep it clean: just use prob_optimal as the score.
    
    scores = prob_optimal
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.perf_counter() - t0
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
