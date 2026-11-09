"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each point.
    
    Returns:
    --------
    scores : array-like, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    t_start = time.time()
    MAX_TIME = 280.0  # Safety margin before 300s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score for Expected Improvement
    diff = f_best - mu
    z = diff / sigma_safe

    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50, 50)

    # Expected Improvement
    ei = diff * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # EI should be non-negative; clip small negatives from numerical error
    ei = np.maximum(ei, 0.0)

    # Probability of Improvement (PI): P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z_clipped)
    pi = np.clip(pi, 0.0, 1.0)

    # Cost-aware scoring: combine EI and PI with cost normalization
    # Use cost^0.5 in denominator to balance exploration efficiency
    cost_denom = np.sqrt(cost_safe)

    # Weighted combination: EI/cost + PI/sqrt(cost)
    # EI gives exploitation signal, PI gives smoother exploration
    score_ei = ei / cost_denom
    score_pi = pi / cost_denom

    # Adaptive weighting: when f_best is far from 0, weight PI more for exploration
    # When f_best is close to 0, weight EI more for exploitation
    f_best_abs = abs(f_best)
    # Transition from exploration (PI-heavy) to exploitation (EI-heavy)
    w_ei = 0.5 + 0.5 * np.exp(-f_best_abs / 2.0)
    w_pi = 1.0 - w_ei

    score = w_ei * score_ei + w_pi * score_pi

    # Add a small uncertainty bonus to encourage exploration in high-uncertainty regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        uncertainty_bonus = 0.05 * (sigma_safe / sigma_max) / cost_denom
        score += uncertainty_bonus

    # Check time deadline
    elapsed = time.time() - t_start
    if elapsed > MAX_TIME:
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
