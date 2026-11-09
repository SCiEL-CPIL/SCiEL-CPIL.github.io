import time
import numpy as np
from scipy.stats import norm

def _norm_cdf(z):
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -10, 10)
    return norm.cdf(z)

def _norm_ppf(q):
    """Inverse CDF (percent point function) for standard normal."""
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return norm.ppf(q)

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)

    # Quantile-based acquisition: maximize expected reduction in 95th percentile
    # of the posterior predictive distribution
    # This is equivalent to minimizing the upper confidence bound (UCB) of the minimum
    # with an implicit balance between exploitation and exploration
    
    # The 95th percentile of the predictive distribution at each point
    # is mu + 1.645 * sigma (for standard normal, 95th percentile ≈ 1.645)
    z_95 = _norm_ppf(0.95)  # ≈ 1.645
    
    # Upper confidence bound of the minimum (we want to minimize this)
    # Lower UCB is better, so we negate it for maximization
    ucb = mu + z_95 * sigma
    
    # Cost-aware: penalize expensive points
    # Use a multiplicative penalty that becomes stronger as cost increases
    # This naturally balances exploration (high sigma) with cost efficiency
    cost_penalty = 1.0 / (1.0 + 0.5 * cost)
    
    # Score: higher is better (we want low UCB and low cost)
    score = -ucb * cost_penalty
    
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety check for deadline
    if time.time() > deadline:
        return score
        
    return score
