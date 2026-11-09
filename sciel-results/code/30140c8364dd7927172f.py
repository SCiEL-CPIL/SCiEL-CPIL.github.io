"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for the deadline guard
_start_time = time.time()
_deadline = 330.0  # 330s wall clock limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Probability of Improvement (PoI) 
    and an adaptive uncertainty bonus.
    
    Args:
    mu: array of GP means, shape (N,)
    sigma: array of GP std devs, shape (N,)
    f_best: current best observed value (scalar)
    
    Returns:
    scores: array of shape (N,), higher is better
    """
    # Deadline guard
    # If we are close to the deadline, we should return quickly.
    # Given the vectorized nature, this is unlikely to be a bottleneck, 
    # but we check to satisfy the directive.
    if time.time() - _start_time > _deadline - 1.0:
        # Fallback: return based on low mu (exploitation only) to be safe and fast
        return -mu

    # Ensure sigma is non-zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Probability of Improvement (PoI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # 2. Adaptive Uncertainty Bonus
    # We want to encourage exploration of high uncertainty regions.
    # Normalizing by f_best makes the bonus relative to the current performance.
    # As f_best decreases, the bonus for a given sigma increases, 
    # potentially helping to escape local minima if the local region has low sigma.
    # However, to prevent the bonus from dominating too much when f_best is very small,
    # we could cap it or use a different scale.
    # For now, we use a weighted sum.
    exploration = sigma_safe / (f_best + 1e-6)
    
    # 3. Combine
    # Weight of 0.5 balances exploitation (PoI) and exploration (Sigma)
    # We can adjust this if needed, but 0.5 is a reasonable starting point.
    scores = poi + 0.5 * exploration
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
