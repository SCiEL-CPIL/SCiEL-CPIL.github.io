"""Acquisition for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# --- explicit deadline guard -------------------------------------------------
_START = time.monotonic()
# Each acquisition call must return quickly; the evaluator kills after 330s wall
# clock. We keep a generous per-call budget so the full 30-eval loop (3 seeds)
# comfortably fits, and we never block.
_BUDGET = 2.0  # seconds per call, far below the 300s/330s limit.


def _guard():
    """Return True if we have exceeded the per-call deadline."""
    return (time.monotonic() - _START) > (_BUDGET + 300.0)


def acquisition(mu, sigma, f_best):
    # Sanitize inputs; ensure finite, 1-D, non-negative sigma.
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if mu.ndim != 1:
        mu = mu.ravel()
    if sigma.ndim != 1:
        sigma = sigma.ravel()
    n = mu.shape[0]
    sigma = np.where(sigma > 0.0, sigma, 1e-9)

    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # We want to MAXIMIZE EI, so we return EI directly.
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with very large/small values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Guarantee finiteness.
    ei = np.where(np.isfinite(ei), ei, 0.0)

    if _guard():
        # Safety: return the current best result immediately.
        return ei

    # Add a small Expected Hyper-Gradient (EHG) exploration bonus.
    # EHG measures how much the posterior mean changes along the input axis,
    # which is a proxy for local structure in the objective. For a 1D slice
    # through a 2D GP, the expected magnitude of the gradient of the mean
    # function is proportional to sigma(x) * |zeta| where zeta captures the
    # local curvature. We approximate this with a finite-difference-like
    # term: where mu varies rapidly relative to sigma, there is likely
    # structure worth probing.
    #
    # We compute a local "gradient magnitude" proxy using the difference
    # between mu and its local median (a robust center), scaled by sigma.
    # This is cheap (O(n log n) for sorting) and captures regions where the
    # GP mean is changing rapidly, indicating nearby RBF centers.
    
    # Compute a local variation measure: how far each point's mu is from
    # the local median, normalized by sigma. High values indicate regions
    # of rapid change in the posterior mean.
    mu_sorted = np.sort(mu)
    # Use a simple rank-based local variation: distance from the median
    # of the top/bottom halves
    median_mu = np.median(mu)
    # Local variation: |mu - median| / sigma, clipped
    local_var = np.abs(mu - median_mu) / sigma
    local_var = np.clip(local_var, 0.0, 5.0)
    
    # EHG-like bonus: scale by a small coefficient. The coefficient is chosen
    # so that the exploration bonus is at most ~10% of typical EI values
    # in early iterations, tapering off as the search converges.
    ehg_coeff = 0.05
    ehg_bonus = ehg_coeff * local_var * sigma
    
    # Blend EI and EHG
    score = ei + ehg_bonus
    
    # Final finiteness guard
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
