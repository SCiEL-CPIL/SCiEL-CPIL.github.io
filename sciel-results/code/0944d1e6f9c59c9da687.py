"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost,
    with a small exploration term to balance exploitation and exploration
    under a tight budget.
    """
    start_time = time.time()
    deadline = 290.0  # Safety margin for 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle edge cases: avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe

    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z_clipped = np.clip(z, -50, 50)

    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)

    ei = (f_best - mu) * cdf_val + sigma * pdf_val

    # Cost-aware score: EI per unit cost
    score_ei = ei / cost_safe

    # Exploration term: sigma per unit cost
    score_unc = sigma / cost_safe

    # Weighted combination: balance exploitation and exploration
    # With a tight budget, exploration is important, but we still want to
    # exploit known good regions.
    scores = 0.7 * score_ei + 0.3 * score_unc

    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        # Return the EI scores as a fallback (they are already computed)
        return score_ei

    return scores
# EVOLVE-BLOCK-END
