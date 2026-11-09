import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Strategy: 
    Uses a modified Lower Confidence Bound (LCB) approach.
    Standard LCB: mu - beta * sigma
    Cost-aware: (mu - beta * sigma) / cost  (Minimizing this value is equivalent to maximizing the score)
    
    However, since we return scores where HIGHER is better, we return:
    Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    Key Structural Change:
    1. Adaptive Beta: Beta is not constant. It scales with the 'regret' (f_best - 0).
       Since Rastrigin has a minimum of 0, a high f_best indicates we are far from the optimum.
       We increase exploration (beta) when f_best is high to escape local minima.
    2. Cost Normalization: We divide by the cost to prefer cheap points that have good potential.
    3. Safety: Ensures finite outputs.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fallback: Pure exploitation per cost
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        return -mu / (cost + 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # f_best is the lowest observed value (scalar)
    if np.isscalar(f_best):
        fb = float(f_best)
    else:
        fb = float(np.min(f_best))

    # Rastrigin 2D on [-5.12, 5.12]^2 has a minimum of 0.
    # The maximum possible value is roughly 2 * (5.12^2 + 10) + 10*2 ~ 600.
    # If f_best is high, we are likely in a bad region or haven't found the global min.
    # We want to encourage exploration when f_best is far from 0.
    
    # Calculate a dynamic beta.
    # If f_best is close to 0, beta should be small (exploit).
    # If f_best is large (e.g., 50), beta should be large (explore).
    # Use a softplus or linear scaling clipped to avoid exploding sigma.
    # Let's use a scale factor that ramps up.
    # beta = base_beta + scale * max(0, f_best - threshold)
    
    # Threshold: Once we get reasonably close (e.g., < 10), reduce exploration.
    # Base beta: 1.0 (standard UCB)
    # Max beta: 5.0 (very aggressive exploration)
    
    regret = max(0.0, fb)  # fb is always >= 0 for rastrigin
    
    # Linear ramp from 1.0 to 5.0 as regret goes from 0 to 50
    beta = 1.0 + np.clip(regret / 50.0, 0.0, 1.0) * 4.0
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the "potential improvement" per cost.
    # Improvement potential ~ (f_best - (mu - beta*sigma)) ? No, LCB is an estimate of the min.
    # If we pick the point with the lowest LCB, we are betting on a low value.
    # Score should be high for points with low LCB.
    # Score = -LCB / cost = (beta * sigma - mu) / cost
    
    lcb = mu - beta * sigma
    score = -lcb / cost
    
    # Handle numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
