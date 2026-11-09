import numpy as np
import time

# ---- Deadline guard ----
_START_TIME = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 20.0

# ---- Constants ----
DIM = 4
RANGE_LOW = -4.0
RANGE_HIGH = 5.0
SPAN = RANGE_HIGH - RANGE_LOW

def acquisition(mu, sigma, f_best, cost):
    """
    Bayesian Regret Minimization acquisition function.
    
    Approximates expected simple regret reduction per unit cost:
    E[min(f(x), f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)
    
    Score = (f_best - E[min(f(x), f_best)]) / cost
          = 0.5 * max(0, f_best - mu - 0.5*sigma) / cost
    
    This directly targets the scoring metric (lowest observed value) by
    estimating how much we expect to improve f_best at each candidate,
    normalized by evaluation cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1e-9)
    
    # Expected improvement in f_best: 0.5 * max(0, f_best - mu - 0.5*sigma)
    # The 0.5*sigma term accounts for uncertainty: when sigma is large,
    # we're less confident about the improvement, so we discount it.
    expected_improvement = 0.5 * np.maximum(0.0, f_best - mu - 0.5 * sigma)
    
    # Cost-normalized score
    score = expected_improvement / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
