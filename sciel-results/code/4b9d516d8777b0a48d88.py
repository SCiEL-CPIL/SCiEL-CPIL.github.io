"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization.
    Uses a structural combination of:
    - Expected Improvement (EI) for exploitation
    - Adaptive UCB exploration term
    - Logarithmic cost penalty (softer than linear division)
    """
    t_start = time.time()
    deadline = 325.0  # 5 second safety margin from 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n == 0:
        return np.array([])

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive exploration: beta decreases as total cost increases
    # Total cost spent approximated by sum of costs of evaluated points
    # We use the mean cost as a proxy for how much budget has been used
    # Early: high beta (explore), Late: low beta (exploit)
    # Cost budget is 30, initial 5 points cost ~1.5 each = 7.5, so ~22.5 remaining
    # Use total_cost as a fraction of budget
    total_cost = np.mean(cost) * n  # rough proxy; actual total is tracked externally
    # Better: use the spread of costs to infer stage
    # Simpler: use a fixed schedule based on the fact that we have ~20 evaluations max
    # Since we don't know the eval count directly, use sigma magnitude as proxy
    # When sigma is large overall, we're early; when small, we're late
    mean_sigma = np.mean(sigma_safe)
    # Normalize: if mean_sigma > 1, early; if < 0.1, late
    # Beta from 3.0 (early) to 0.5 (late)
    beta = 0.5 + 2.5 * np.exp(-mean_sigma * 5.0)

    # UCB exploration term
    ucb = beta * sigma_safe

    # Combine EI and UCB
    acquisition_raw = ei + ucb

    # Cost-aware penalty: use log(1 + cost) to soften the penalty
    # This gives less extreme penalty differences than linear division
    # Scale: cost ranges [1, 2], so log(1+cost) ranges [log(2), log(3)] ≈ [0.69, 1.10]
    cost_penalty = np.log(1.0 + cost_safe)
    # Normalize cost penalty to have mean 1 for stability
    cost_penalty_mean = np.mean(cost_penalty)
    cost_penalty_norm = cost_penalty / cost_penalty_mean

    # Final score: acquisition divided by normalized cost penalty
    scores = acquisition_raw / cost_penalty_norm

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    if time.time() - t_start > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
