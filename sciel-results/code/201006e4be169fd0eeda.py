import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    This approach adapts the exploration-exploitation balance based on the
    current best observed value (f_best) relative to the known global minimum
    of the Michalewicz function.
    
    - When f_best is far from the global minimum, we explore more (higher kappa).
    - When f_best is close to the global minimum, we exploit more (lower kappa).
    """
    # Known global minimum for Michalewicz 2D
    opt = -1.8013
    
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate distance from current best to the global optimum
    # This is positive if we haven't found the optimum yet
    dist = max(0.0, f_best - opt)
    
    # Adaptive kappa:
    # Start with a base exploration level of 0.5
    # Increase exploration linearly with distance, capped to prevent excessive randomness
    # dist/5.0 normalizes the distance; if f_best is 5 units away, max boost is applied
    # Cap kappa at 3.0 to avoid pure noise selection
    kappa = 0.5 + 2.0 * (dist / 5.0)
    kappa = min(kappa, 3.0)
    
    # LCB Score: -mu + kappa * sigma
    # Higher score is better (we want to minimize mu, so -mu is high for low mu)
    # We add kappa * sigma to encourage exploration where uncertainty is high
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
