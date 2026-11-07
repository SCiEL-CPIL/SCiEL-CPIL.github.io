"""Cost-aware acquisition for Powell 4D BO using Probability of Improvement with inverse-cost weighting.

Structural change: Switch from Expected Improvement (EI) with a Gaussian decay cost penalty to
Probability of Improvement (PI) with linear inverse-cost weighting (PI / cost). PI naturally
saturates at 1.0 for high-confidence improvements and decays smoothly with uncertainty,
providing a bounded, well-behaved signal. Dividing by cost (rather than multiplying by a
decaying penalty) creates a linear efficiency trade-off that avoids the over-suppression of
moderately expensive points seen with Gaussian penalties, while still strongly favoring
cheap evaluations. This is a standard cost-aware BO formulation that directly optimizes
improvement probability per unit resource.
"""
import numpy as np
from scipy.stats import norm
import time

# ------------------------------- deadline guard ---------------------------------
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 15.0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


# ------------------------------- public entry point ------------------------------
def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement acquisition function with linear inverse-cost weighting.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if _time_left() < 1.0:
        # Emergency fallback: just return negative mu (exploitation)
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)

    # Probability of Improvement for minimization:
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    pi = norm.cdf(z)
    pi = np.clip(pi, 1e-10, 1.0)  # Keep in (0, 1] for numerical stability

    # Structural Change: Use Probability of Improvement per unit Cost (PI/Cost)
    # This is a standard cost-aware acquisition function that optimizes
    # improvement probability efficiency rather than raw improvement magnitude.
    # It linearly trades off cost against improvement confidence.
    # Since cost is in [1, 2], this scales PI by 0.5 to 1.0.
    
    # Ensure cost is strictly positive for division
    cost_safe = np.maximum(cost, 1e-6)
    
    score = pi / cost_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
