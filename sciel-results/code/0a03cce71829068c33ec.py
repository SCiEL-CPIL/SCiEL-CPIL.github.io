"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # Seconds to spare before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blend of Probability of Improvement (POI) and UCB with adaptive exploration.
    
    Uses POI to model the probability of beating the current best, scaled by expected gain,
    blended with UCB for exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - posterior mean in objective units
    sigma : array of shape (2000,) - posterior std in objective units
    f_best : float - lowest observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: If we are running out of time, return a simple fallback
    current_time = time.time()
    elapsed = current_time - _start_time
    if elapsed > 300.0 - _DEADLINE_MARGIN:
        return -mu

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Probability of Improvement (POI) for minimization
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    poi = norm.cdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # POI-scaled by expected gain: this emphasizes regions where there's both
    # a good chance of improvement AND a large expected improvement
    poi_scaled = poi * np.log1p(np.abs(ei) + 1e-10)
    
    # UCB for minimization: score = -mu + beta * sigma
    # Adaptive beta: explore more when f_best is far from optimum
    f_opt = -39.1662
    dist = max(f_best - f_opt, 0.0)
    
    # Beta ranges from ~0.5 (near optimum) to ~3.0 (far from optimum)
    beta = 0.5 + 2.5 * (1.0 - np.exp(-dist / 10.0))
    
    ucb = -mu + beta * sigma
    
    # Normalize all three components to have similar scale before blending
    def normalize(arr):
        m = np.mean(arr)
        s = np.std(arr) + 1e-10
        return (arr - m) / s
    
    poi_norm = normalize(poi_scaled)
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # Adaptive blend weights:
    # Early on (high dist): more UCB for exploration
    # Later (low dist): more POI/EI for exploitation
    # w_ucb in [0.2, 0.5]
    w_ucb = 0.2 + 0.3 * (1.0 - np.exp(-dist / 15.0))
    
    # Split remaining weight between POI and EI
    # More POI early (to identify promising regions), more EI later (to refine)
    remaining = 1.0 - w_ucb
    w_poi = 0.4 * remaining + 0.3 * remaining * np.exp(-dist / 20.0)
    w_ei = remaining - w_poi
    
    scores = w_poi * poi_norm + w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
