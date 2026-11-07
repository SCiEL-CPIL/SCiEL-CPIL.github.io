import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    Uses an adaptive LCB with cost normalization.
    """
    # Track time to avoid timeout
    t_start = time.time()
    deadline = t_start + 290.0  # safety margin before 300s limit

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma to be non-negative and finite
    sigma = np.clip(sigma, 0.0, None)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)

    # Ackley 2D: min 0, max ~29
    f_max_ackley = 29.0
    f_ratio = np.clip(f_best / f_max_ackley, 0.0, 1.0)

    # Adaptive beta: start exploratory, become exploitative
    beta = 1.0 + 2.0 * f_ratio

    # LCB for minimization
    lcb = -mu + beta * sigma

    # Cost-aware normalization
    cost_weight = 1.0 / np.sqrt(cost)

    # Final score
    score = lcb * cost_weight

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -1e6)

    return score
