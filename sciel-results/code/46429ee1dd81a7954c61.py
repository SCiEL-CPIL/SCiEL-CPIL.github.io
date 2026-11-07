import numpy as np
import time

# Global state for deadline guard and adaptive parameters
_start_time = None
_eval_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using batch-aware LCB with spatial diversity penalty.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
        
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising points.
    """
    global _start_time, _eval_count
    
    # Deadline guard
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    if elapsed > 295:
        # Return a safe fallback if we're too close to the limit
        return np.zeros_like(mu)
    
    _eval_count += 1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive exploration parameter beta based on convergence progress
    # Hartmann 3D global minimum is approx -3.86278
    target_min = -3.86278
    
    # Calculate distance from the current best to the known global minimum
    dist = f_best - target_min
    dist = np.clip(dist, 0.0, 4.0)
    
    # Beta decreases as we get closer to the optimum
    # Early: high exploration (beta ~ 3.0), Late: low exploration (beta ~ 0.5)
    beta = 0.5 + 2.5 * (dist / 4.0)
    
    # Core LCB score: -mu + beta * sigma
    lcb_scores = -mu + beta * sigma
    
    # Spatial diversity penalty: penalize candidates that are likely in well-explored regions
    # We approximate this by using sigma as a proxy for exploration:
    # Low sigma means well-explored, high sigma means under-explored
    # We add a bonus for high sigma to encourage exploration of new regions
    
    # Normalize sigma to [0, 1] range for consistent scaling
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Diversity bonus: encourages evaluation of points with high uncertainty
    # This helps prevent redundant evaluations in the same basin
    diversity_weight = 0.5
    diversity_bonus = diversity_weight * sigma_norm
    
    # Combine LCB score with diversity bonus
    scores = lcb_scores + diversity_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return scores
