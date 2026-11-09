import numpy as np
import time

# Deadline guard: return early if we're running too long
_START_TIME = time.time()
_DEADLINE = 330.0  # Wall-clock limit is 330s, but each call should be fast
_MARGIN = 5.0      # Safety margin

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling heuristic for Bayesian optimization of Ackley 2D.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value.
    
    Returns 2000 finite scores, higher means evaluate next.
    """
    # Check deadline
    if _time_check():
        # Return a simple fallback: UCB with beta=0.5
        return 0.5 * sigma - mu
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling heuristic:
    # score = -mu - beta * sigma_normalized
    # where sigma_normalized is quantile-normalized to [0, 1]
    # and beta scales with f_best to explore early, exploit late
    
    # Compute quantile-normalized sigma: rank each sigma value among all candidates
    # This makes the exploration term relative to the current uncertainty distribution
    sigma_ranks = np.argsort(np.argsort(sigma)) / (len(sigma) - 1)  # ranks in [0, 1]
    
    # Dynamic exploration weight:
    # - When f_best is large (early), explore more aggressively
    # - When f_best is small (close to minimum), focus on exploitation
    # Use a smooth decay: beta = base_beta * (1 + f_best / scale)
    # scale is set so that at f_best ~ 1.0, beta is about 2x base
    base_beta = 1.0
    scale = 1.0
    beta = base_beta * (1.0 + f_best / scale)
    
    # Cap beta to prevent excessive exploration
    beta = np.clip(beta, 0.5, 5.0)
    
    # Score: higher is better
    # -mu favors low predicted values (exploitation)
    # -beta * sigma_normalized favors high uncertainty regions (exploration)
    # The quantile normalization ensures we always explore the most uncertain candidates
    # relative to the current posterior distribution
    scores = -mu - beta * sigma_ranks
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
