"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback: just pick the one with lowest mu / cost
        # to avoid crashing, though the run is likely over.
        return -mu / (cost + 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive LCB: use a dynamic beta that decreases as f_best improves
    # f_best starts high (ackley min is 0), so we want more exploration early
    # and less exploration (more exploitation) as f_best approaches 0.
    
    # Scale beta based on f_best: when f_best is large, beta is larger (more exploration)
    # When f_best is close to 0, beta is smaller (more exploitation)
    # Use a soft decay: beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    beta_max = 2.0  # Maximum exploration (early)
    beta_min = 0.5   # Minimum exploration (late, near optimum)
    scale = 1.0      # Decay scale
    
    # f_best is the lowest observed value; for ackley, this starts around 1-3 and goes to 0
    # We want high beta when f_best is high, low beta when f_best is low
    beta = beta_min + (beta_max - beta_min) * np.exp(-f_best / scale)
    
    # LCB = mu - beta * sigma (we want to minimize, so lower LCB is better)
    # Acquisition score should be higher for more promising points
    # So we use: score = -LCB / cost = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    lcb = mu - beta * sigma_safe
    score = -lcb / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
