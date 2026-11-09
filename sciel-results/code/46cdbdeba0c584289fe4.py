"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Lowest observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    t_start = time.time()
    DEADLINE = 295.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Check deadline
    if time.time() - t_start > DEADLINE:
        return np.zeros_like(mu)

    # Compute z-score for minimization
    z = (f_best - mu) / sigma_safe

    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)

    # Expected Improvement for minimization
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z

    # Check deadline
    if time.time() - t_start > DEADLINE:
        return ei / cost_safe

    # Use a pure cost-normalized Expected Improvement with a small
    # exploration bonus that is also cost-normalized.
    # The exploration bonus uses a tighter, bounded form to avoid
    # over-exploration while still maintaining diversity.
    # 
    # Key structural change: instead of a fixed blend of EI and UCB,
    # we use a "probabilistic cost-normalized improvement" that
    # accounts for the variance in cost-normalized scores.
    #
    # The score is: EI / cost + lambda * sigma / cost
    # where lambda is chosen to balance exploration.
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe

    # Cost-normalized exploration term
    # Use a smaller, bounded exploration coefficient
    # The coefficient adapts: when f_best is close to 0 (near optimum),
    # reduce exploration; when far, increase it.
    # f_best starts high (powell min is 0, so f_best > 0 initially)
    # As f_best approaches 0, exploration should decrease.
    # Scale: when f_best ~ 10, beta ~ 0.3; when f_best ~ 0.1, beta ~ 0.05
    beta = 0.3 * min(f_best, 1.0) + 0.05
    
    ucb = beta * sigma_safe
    ucb_cost = ucb / cost_safe

    # Combined score
    scores = ei_cost + ucb_cost

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() - t_start > DEADLINE:
        return scores

    return scores
# EVOLVE-BLOCK-END
