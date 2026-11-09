import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization) of Weierstrass 1D.
    
    Blends Thompson Sampling (for exploration diversity) with Expected Improvement
    (for exploitation precision), with adaptive weighting based on posterior
    uncertainty relative to current best.
    """
    start_time = time.time()
    deadline = 330.0
    safety_margin = 5.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)

    # Handle sigma ~ 0 cases to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute Thompson Sampling score
    # Use a deterministic seed derived from the posterior state to ensure
    # reproducibility without interfering with the evaluator's RNG
    sum_mu = np.sum(mu)
    sum_sigma = np.sum(sigma)
    seed = int(abs(sum_mu) * 12345 + abs(f_best) * 7777 + abs(sum_sigma) * 3333) % (2**32)
    rng = np.random.default_rng(seed)

    # Draw samples from the posterior
    ts_samples = mu + sigma_safe * rng.standard_normal(n)

    # For minimization, lower samples are better, so score = -ts_samples
    ts_score = -ts_samples

    # Adaptive weighting:
    # When f_best is close to 0 (good), exploit more (higher EI weight)
    # When f_best is far from 0 (bad), explore more (higher TS weight)
    f_best_clamped = np.clip(f_best, 0.0, 2.0)

    # EI weight goes from ~0.2 (explore) to ~0.8 (exploit) as f_best improves
    weight_ei = 0.2 + 0.6 * np.exp(-1.5 * f_best_clamped)
    weight_ts = 1.0 - weight_ei

    # Scale EI to be comparable to TS score magnitude
    # EI is in objective units; TS score is also in objective units
    # Use the median absolute deviation of TS score to normalize EI contribution
    ts_scale = np.median(np.abs(ts_score)) + 1e-12
    ei_scaled = ei * (ts_scale / (np.median(ei[ei > 0]) + 1e-12)) if np.any(ei > 0) else ei

    # Blend: TS provides the base exploration signal, EI adds exploitation bonus
    score = weight_ts * ts_score + weight_ei * ei_scaled

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    # Check time
    if time.time() > deadline - safety_margin:
        return score

    return score
