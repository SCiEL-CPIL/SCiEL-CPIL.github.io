import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge case where f_best is very small (near convergence)
    if f_best <= 1e-9:
        return -mu
    
    # Ensure sigma is non-negative and safe for normalization
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Quantile normalization of sigma (rank-based)
    # This maps sigma to [0, 1] based on its relative rank among candidates.
    # This is robust to the absolute scale of sigma and ensures that
    # the exploration term is always in a consistent range [0, 1] * beta.
    n = len(sigma_safe)
    if n > 1:
        ranks = np.argsort(np.argsort(sigma_safe))
        sigma_norm = ranks / (n - 1)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Dynamic exploration weight
    # We scale exploration with f_best.
    # When f_best is large (early), we explore more.
    # When f_best is small (late), we exploit more.
    # The coefficient 1.0 is a balance:
    # - If beta is too high, we wander.
    # - If beta is too low, we get stuck in local minima.
    # Given the Ackley function's multimodality and the limited budget (30 evals),
    # a moderate exploration weight is key.
    beta = 1.0 * f_best
    
    # Score: Exploitation (-mu) + Exploration (beta * sigma_norm)
    # Higher score means "evaluate this next".
    scores = -mu + beta * sigma_norm
    
    # Ensure finite scores
    # Replace any NaN or Inf with a very low value to avoid crashing the loop
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
