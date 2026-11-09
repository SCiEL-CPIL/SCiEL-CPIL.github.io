# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# ---- module-level deadline guard ----
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # safety margin before 330s kill


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on the
    three-hump camel 2D. Combines a cost-adjusted Expected Improvement
    (EI) term with a Thompson-sampling-style stochastic exploration bonus,
    both normalized by evaluation cost.

    Parameters
    ----------
    mu     : (2000,) GP mean predictions (objective units)
    sigma  : (2000,) GP standard deviation (objective units)
    f_best : float  best (lowest) observed value
    cost   : (2000,) cost of evaluating each candidate

    Returns
    -------
    scores : (2000,) finite, higher = evaluate next
    """
    # Deadline guard: if we're running out of time, return a fast fallback
    if _check_deadline():
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-10)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-10)
        mu = np.asarray(mu, dtype=np.float64)
        # Simple fallback: cost-adjusted (f_best - mu + sigma)
        return (float(f_best) - mu + sigma_safe) / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    f_best = float(f_best)

    # ---- Expected Improvement (EI) for minimization ----
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -12.0, 12.0)
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = delta * cdf + sigma_safe * pdf

    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)

    # ---- Cost normalization ----
    # Normalize cost so that the mean cost is 1; cheaper points get boosted
    cmean = float(np.mean(cost_safe))
    if cmean <= 0.0:
        cmean = 1.0
    cost_norm = cost_safe / cmean

    # ---- Cost-adjusted EI ----
    ei_adjusted = ei / cost_norm

    # ---- Stochastic exploration bonus (Thompson-sampling inspired) ----
    # Sample from the posterior at each candidate: x ~ N(mu, sigma^2)
    # This naturally balances exploration (high sigma) and exploitation (low mu)
    # Use a deterministic pseudo-random seed based on the candidate index for reproducibility
    # within a single call, but vary across calls to avoid deterministic bias
    n = mu.size
    # Use a time-based seed for stochasticity across calls
    seed = int(time.time() * 1000) % (2**31)
    rng = np.random.default_rng(seed)
    # Sample from the posterior
    samples = mu + sigma_safe * rng.standard_normal(n)
    # The "improvement" from sampling: how much better is the sample than f_best?
    # We want to prefer candidates where the sample is likely to beat f_best
    # Use the negative sample (lower is better for minimization)
    # Scale by 1/cost for cost-awareness
    beta_explore = 0.3
    expl_bonus = beta_explore * (-samples) / cost_norm

    # ---- Combine: cost-adjusted EI + exploration bonus ----
    scores = ei_adjusted + expl_bonus

    # ---- Guard finiteness ----
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e9)

    return scores
# EVOLVE-BLOCK-END
