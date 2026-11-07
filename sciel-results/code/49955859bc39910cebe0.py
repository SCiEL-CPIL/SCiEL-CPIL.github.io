"""Cost-aware acquisition: adaptive EI/UCB blend with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level state for deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.95  # return fallback after 95% of 300s wall-clock


def _elapsed_ok():
    """Return True if we are within the deadline; initialize on first call."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) < 300.0 * _DEADLINE_FRACTION


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: return a safe finite fallback (exploit best mu per cost)
    if not _elapsed_ok():
        safe_cost = np.maximum(cost, 1e-9)
        return -mu / safe_cost

    # Guard against degenerate inputs
    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected improvement (minimization): EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is non-negative; floor to avoid negative values from numerical noise
    ei = np.maximum(ei, 0.0)

    # UCB: mu - kappa * sigma  (lower is better, so we maximize -mu + kappa*sigma)
    # Use a moderate kappa; UCB naturally explores high-sigma regions.
    kappa = 1.0
    ucb = -mu + kappa * sigma_safe  # higher = more promising to explore

    # Adaptive blend: weight EI heavily near the end of budget, UCB early.
    # Normalize cost to [0,1] over the observed candidates; the budget is ~30.
    # Use a smooth ramp: as mean cost approaches 30, shift to exploitation.
    mean_cost = float(np.mean(cost_safe)) if n > 0 else 0.0
    # progress in [0,1]: 0 at start, 1 near budget 30
    progress = np.clip(mean_cost / 30.0, 0.0, 1.0)
    # EI weight: small early, large late. UCB weight: large early, small late.
    # Use a quadratic ramp so the transition is smooth.
    w_ei = 0.2 + 0.8 * (progress ** 2)
    w_ucb = 1.0 - w_ei

    # Normalize each component to comparable scale before blending.
    # EI is in objective units; ucb is in objective units too.
    # Scale ucb by a factor to match EI's typical magnitude (EI ~ |f_best - mu|).
    # Use the spread of (f_best - mu) as a reference scale.
    diff = np.maximum(f_best - mu, 0.0)
    ref_scale = float(np.mean(diff)) if n > 0 else 1.0
    ref_scale = max(ref_scale, 1e-6)

    # Combine: blend in a common scale, then make cost-aware.
    blended = w_ei * ei + w_ucb * (ref_scale * ucb)
    # Ensure non-negative contribution so cost-awareness doesn't flip sign oddly
    blended = np.maximum(blended, 0.0)

    # Cost-aware: divide by cost so cheap evaluations are favored per unit cost.
    score = blended / cost_safe

    # Final safety: ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
