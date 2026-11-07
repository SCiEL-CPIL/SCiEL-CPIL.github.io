"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses Thompson Sampling with cost normalization: sample from N(mu, sigma),
    then score = -sample / cost (higher score = lower predicted value per unit cost).
    
    Parameters
    ----------
    mu : array (2000,) GP mean
    sigma : array (2000,) GP std
    f_best : float best observed value
    cost : array (2000,) evaluation cost per candidate
    
    Returns
    -------
    scores : array (2000,) higher = evaluate next
    """
    t0 = time.time()
    DEADLINE = 325.0  # safety margin before 330s wall-clock limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1.0)

    n = len(mu)

    # Thompson Sampling: draw a sample from the posterior predictive N(mu, sigma)
    # The sample represents a plausible value of f(x)
    rng = np.random.default_rng()
    sample = mu + sigma * rng.standard_normal(n)

    # For minimization, we want lower values to be better.
    # Score = -sample / cost: lower sample (better) and lower cost give higher score.
    scores = -sample / cost

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)

    # Check time
    if time.time() - t0 > DEADLINE:
        return scores

    return scores
# EVOLVE-BLOCK-END
