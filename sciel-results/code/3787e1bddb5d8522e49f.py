"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills the program after 330s wall-clock.
# We keep a safety margin and bail out with a valid result if we run long.
_DEADLINE_S = 300.0  # safety margin below the 330s kill
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid EI/UCB acquisition for minimizing Powell 4D.

    Parameters:
        mu: (2000,) GP mean predictions (objective units)
        sigma: (2000,) GP std predictions (objective units)
        f_best: scalar, best observed value (objective units)
        cost: (2000,) evaluation costs (in [1, 2])

    Returns:
        (2000,) acquisition scores (higher = evaluate next), all finite.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: if we are close to the deadline, return a valid finite result
    # based on pure exploitation (lowest mu / cost) so the loop can finish.
    if _elapsed() > _DEADLINE_S:
        out = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(out), out, 0.0)

    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Expected improvement (for minimization)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Adaptive UCB exploration coefficient:
    # scale by global uncertainty relative to mean spread.
    med_sigma = float(np.median(sigma))
    mu_std = float(np.std(mu))
    if mu_std < 1e-12:
        mu_std = 1e-12
    # kappa grows when the model is globally uncertain relative to its confidence
    kappa = 2.0 * (med_sigma / (mu_std + med_sigma + 1e-12))
    # Cap kappa to keep it bounded
    kappa = float(np.clip(kappa, 0.5, 4.0))

    ucb = mu + kappa * sigma

    # Cost-aware: lower cost is better, so subtract a cost-normalized value.
    # For EI (higher is better): divide by cost.
    score_ei = ei / cost
    # For UCB (lower is better for minimization): negate and divide by cost.
    score_ucb = -ucb / cost

    # Blend exploitation and exploration.
    # Use a fixed 50/50 blend; the adaptive kappa already modulates exploration.
    score = 0.5 * score_ei + 0.5 * score_ucb

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
