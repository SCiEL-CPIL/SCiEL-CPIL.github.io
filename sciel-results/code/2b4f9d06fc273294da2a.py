"""Cost-aware acquisition for Powell 4D BO using Probability of Improvement with Cost Penalty.

Structural change: Switch from Expected Improvement (EI) to Probability of Improvement (PI).
PI is bounded and less sensitive to extreme uncertainty (high sigma) compared to EI, which can
lead to excessive exploration in expensive regions. By using PI / Cost, we favor points that
have a high *probability* of beating the current best, weighted by their cost efficiency.
This provides a more stable and focused search trajectory, particularly in the presence of
cost constraints.
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
    Cost-aware Probability of Improvement acquisition function.
    
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
    # PI(x) = P(f(x) < f_best)
    # = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    pi = norm.cdf(z)
    
    # Structural Change: Use Probability of Improvement per unit Cost (PI/Cost)
    # This favors points with a high likelihood of improvement, penalized by cost.
    # Unlike EI, PI does not grow with sigma, preventing "exploration drift" to expensive,
    # highly uncertain regions.
    
    # Ensure cost is strictly positive for division
    cost_safe = np.maximum(cost, 1e-6)
    
    score = pi / cost_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
