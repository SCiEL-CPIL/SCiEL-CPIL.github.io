import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)

    # Sanitize
    sigma = np.maximum(sigma, 1e-9)
    cost_arr = np.maximum(cost_arr, 1e-9)

    # Structural Change: Pure Thompson Sampling
    # Draw a single sample from the posterior for each candidate
    # y ~ N(mu, sigma)
    # Score = -y / cost
    # This treats the objective as a stochastic reward-to-go ratio.
    
    # To balance exploration/exploitation, we can slightly inflate sigma if f_best is still high,
    # but the directive suggests a structural change to TS.
    # Let's stick to standard TS first, maybe with a small adaptive noise if f_best is very high?
    # No, let's keep it pure as per the strategy: "drawing a single random sample from the GP posterior (mu - sigma * z)"
    # Wait, the strategy says "mu - sigma * z"? Usually it's mu + sigma * z.
    # "mu - sigma * z" would be minimizing a sample if we want low values?
    # If we want to minimize y, and score is higher is better, score = -y.
    # If y = mu - sigma * z, then for positive z, y is lower than mu.
    # Standard TS for minimization: sample y ~ N(mu, sigma). Select min(y).
    # Since we return scores (higher is better), we return -y / cost.
    
    # Let's use standard normal sampling.
    z = np.random.normal(size=mu.shape)
    y = mu + sigma * z
    
    # Cost-aware score
    scores = -y / cost_arr
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
