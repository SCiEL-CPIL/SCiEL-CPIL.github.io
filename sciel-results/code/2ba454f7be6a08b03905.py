"""Cost-aware acquisition for BO on hartmann 6d (minimisation)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Saturated Expected Improvement (Hypar-like).
    
    Uses a hyperbolic tangent saturation on the z-score to bound the improvement
    estimate, which can be more robust than linear EI for cost-aware selection.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value (minimisation).
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x) for x in [0,1]^6.
    
    Returns
    -------
    ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    t0 = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-10)

    # z-score for improvement (minimisation: improvement = f_best - mu)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Saturated Improvement: sigma * tanh(z)
    # This bounds the "improvement" contribution to [-sigma, sigma],
    # unlike EI which grows linearly with (f_best - mu).
    # For large z, tanh(z) -> 1, so score ~ sigma / cost.
    # For small z, tanh(z) ~ z, so score ~ (f_best - mu) / cost.
    sat_improve = sigma * np.tanh(z)
    
    # Ensure positive score for improvement (tanh(z) is negative if mu > f_best)
    # We only care about improvement, so we can clip negative values to 0 or use abs?
    # In EI, negative improvements are 0.
    # tanh(z) is negative when z < 0 (mu > f_best).
    # So we should clip to 0.
    sat_improve = np.maximum(sat_improve, 0.0)

    # Cost-aware score: Saturated Improvement / cost^alpha
    # Using alpha = 1.0 for standard cost-awareness
    alpha = 1.0
    cost_penalty = np.power(cost, alpha)
    score = sat_improve / cost_penalty

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
