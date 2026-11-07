"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 285.0

    # Ensure arrays are 1-D numpy arrays of the right type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.size
    if n == 0:
        return np.array([], dtype=np.float64)

    # Floor sigma to avoid division by zero / log of zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # Thompson Sampling: sample y ~ N(mu, sigma^2) for each candidate
    # Use a fixed random state for reproducibility within a call
    rng = np.random.RandomState(42)
    y_samples = mu + sigma_safe * rng.standard_normal(n)

    # For minimization, we want low y. But we also want to account for cost.
    # The acquisition score should be high for points we want to evaluate next.
    # We want to minimize y / cost, so score = - (y / cost) or score = 1 / (y / cost) if y > 0?
    # Since f can be negative (hartmann), y can be negative.
    # A better approach: score = -y / cost_safe.
    # This favors low y (good prediction) and low cost (cheap).
    # However, if y is very negative (large improvement), -y is large positive.
    # If y is positive (worse than 0), -y is negative.
    # We want the highest score to be selected.
    # So score = -y / cost_safe seems reasonable.
    
    # But wait, Thompson Sampling usually samples the function value and picks the argmin (for minimization).
    # The cost-aware version should pick the argmin of y / cost.
    # So we want the acquisition function to return high values for candidates with low y/cost.
    # Thus, score = - (y / cost_safe) works.
    
    scores = -y_samples / cost_safe

    # Check deadline
    if time.time() > deadline:
        return np.zeros_like(mu)

    # Guarantee finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
