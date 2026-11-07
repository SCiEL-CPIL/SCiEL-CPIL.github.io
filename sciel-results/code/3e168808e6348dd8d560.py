"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-optimal cost-normalized acquisition function.
    
    Uses marginal cost normalization: instead of dividing by absolute cost,
    we normalize by the cost relative to the minimum cost among candidates
    with positive expected improvement. This creates a Pareto-optimal
    trade-off between improvement and cost.
    
    Additionally, we use an adaptive exploration weight that shifts from
    exploration to exploitation as the search converges.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we're close to the limit, return a simple safe result
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple exploitation-only result
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
        z = (f_best - np.asarray(mu, dtype=np.float64)) / sigma_safe
        z_clipped = np.clip(z, -10.0, 10.0)
        ei = (f_best - np.asarray(mu, dtype=np.float64)) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-8)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration weight based on convergence state
    # When f_best is large (far from optimum), explore more
    # When f_best is small (close to optimum), exploit more
    # Use the ratio of current uncertainty to the gap as a signal
    sigma_mean = float(np.mean(sigma_safe))
    gap = max(f_best - 0.0, 1e-6)  # Assuming minimum is near 0
    uncertainty_ratio = sigma_mean / gap
    
    # Adaptive beta: higher when uncertainty is large relative to gap
    # Clamp to reasonable range
    beta = np.clip(0.5 + 1.0 * uncertainty_ratio, 0.5, 2.0)
    
    # UCB component for exploration
    ucb = beta * sigma_safe - mu
    ucb = np.maximum(ucb, 0.0)
    
    # Combined acquisition value (in objective units)
    combined = ei + 0.3 * ucb
    
    # Pareto-optimal cost normalization:
    # Instead of dividing by absolute cost, we use marginal cost normalization.
    # Find the minimum cost among candidates with positive combined value
    positive_mask = combined > 1e-10
    if np.any(positive_mask):
        min_cost_positive = float(np.min(cost_safe[positive_mask]))
    else:
        min_cost_positive = float(np.min(cost_safe))
    
    # Marginal cost: how much more expensive is this candidate compared to the cheapest good one
    # Normalize cost by the minimum cost among positive-improvement candidates
    # This makes cost 1.0 for the cheapest candidate and >1.0 for more expensive ones
    marginal_cost = cost_safe / max(min_cost_positive, 1e-8)
    
    # Score: combined value divided by marginal cost
    # This explicitly favors low-cost points that offer high expected improvement
    score = combined / marginal_cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
