import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        # Fallback: Fixed LCB
        return -mu - 2.0 * np.maximum(sigma, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration-exploitation trade-off
    # Weierstrass function minimum is 0.
    # f_best is the lowest observed value.
    # We map f_best to a beta parameter for the LCB.
    # High f_best (far from min) -> High beta (more exploration).
    # Low f_best (close to min) -> Low beta (more exploitation).
    
    # Clip f_best to [0, 1] to ensure stable beta calculation.
    # If f_best > 1, it's effectively "very far", so beta_max.
    # If f_best < 0, it's "at min" (theoretical), so beta_min.
    f_best_clamped = np.clip(f_best, 0.0, 1.0)
    
    beta_min = 0.5
    beta_max = 5.0
    
    # Linear mapping from f_best [0, 1] to beta [beta_min, beta_max]
    # beta = beta_min + (beta_max - beta_min) * f_best_clamped
    beta = beta_min + (beta_max - beta_min) * f_best_clamped
    
    # LCB acquisition function for minimization:
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We select the point that minimizes LCB, which is equivalent to maximizing -LCB(x).
    # Score = -mu(x) + beta * sigma(x)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
