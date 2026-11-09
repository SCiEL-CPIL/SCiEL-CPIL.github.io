import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) with
    median-normalized sigma to stabilize exploration across the domain.
    
    This structural change (normalization) addresses potential scale
    artifacts in the GP posterior std and focuses exploration relative
    to the typical uncertainty magnitude.
    """
    # Deadline guard: return a simple fallback if running out of time
    if _time_check():
        # Fallback: pure exploitation-like score to finish quickly
        return -mu

    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    
    # Structural Change: Normalize sigma by its median.
    # This makes the exploration term 'beta * sigma' relative to the
    # typical uncertainty level of the current GP posterior, preventing
    # regions with naturally higher/lower GP variance (due to kernel
    # effects or data sparsity) from being overly favored or penalized
    # solely due to scale.
    med_sigma = np.median(sigma)
    if med_sigma < 1e-12:
        med_sigma = 1e-12
    sigma_norm = sigma / med_sigma
    
    # Adaptive Beta:
    # We want to explore more when f_best is high (early) and exploit
    # more when f_best is low (late).
    # Using a logarithmic scale for f_best to handle the wide range of
    # Ackley values (0 to ~20) more gracefully than linear clipping.
    # log(1 + f_best) is 0 at f_best=0, ~3 at f_best=20.
    # We map this to a beta range.
    # Early (f_best ~ 5): log(6) ~ 1.8. Beta ~ 1.0 + 1.8 = 2.8?
    # Let's calibrate:
    # We want beta ~ 2.0 at start, ~ 0.5 at end.
    # beta = 0.5 + 1.5 * (log(1+f_best) / log(1+5))
    # If f_best=5, ratio=1, beta=2.0.
    # If f_best=0, ratio=0, beta=0.5.
    # If f_best=20, log(21)/log(6) ~ 3/1.8 ~ 1.6. Clip to 1.
    
    f_best_safe = max(f_best, 0.0)
    ratio = np.log(1.0 + f_best_safe) / np.log(1.0 + 5.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    
    beta = 0.5 + 1.5 * ratio
    
    # Calculate UCB score
    # Higher score is better. We minimize f, so we prefer low mu and high sigma.
    # score = -mu + beta * sigma_norm
    
    scores = -mu + beta * sigma_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
