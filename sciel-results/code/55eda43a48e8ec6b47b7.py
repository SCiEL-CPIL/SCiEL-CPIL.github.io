"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280
_ACQ_CALL_COUNT = [0]


def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - np_sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    _ACQ_CALL_COUNT[0] += 1

    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        # Fallback: simple cost-aware mean minimization
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Sanitize inputs
    sigma = np.maximum(np_sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Strategy: Probability of Improvement (POI)
    # POI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    # This encourages exploration of regions where there is a significant probability
    # of finding a better solution, but is less conservative than EI.
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in CDF calculation
    # Phi(-10) is approx 0, Phi(10) is approx 1
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate POI
    poi = norm.cdf(z_clipped)
    
    # Cost-aware scoring
    # We want to maximize the probability of improvement per unit cost.
    # Dividing by cost directly rewards cheap points.
    # However, pure POI/cost might be too aggressive on cheap but low-potential points.
    # A slight exploration bonus based on sigma can help, but POI inherently
    # balances exploitation (high z) and exploration (high sigma -> wider spread).
    
    # To further enhance cost-efficiency, we normalize the score.
    # The cost function is 1 + mean(x_scaled), so costs are in [1, 2].
    # Dividing by cost is a strong signal.
    
    score = poi / cost
    
    # Optional: Add a tiny bit of pure uncertainty exploration to prevent
    # complete convergence to the current best if the GP is confident but wrong.
    # However, the finding suggests POI alone is strong. Let's stick to a clean POI/cost
    # but ensure it doesn't become zero if sigma is very small and mu is close to f_best.
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
