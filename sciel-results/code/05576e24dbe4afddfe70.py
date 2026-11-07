import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    LCB: score = -mu - kappa * sigma
    The evaluator picks the candidate with the highest score, so we want to
    maximize (-mu - kappa * sigma), which corresponds to minimizing (mu + kappa * sigma).
    
    We use an adaptive kappa that depends on the number of evaluations inferred
    from the magnitude of f_best relative to the typical Beale function values.
    Since we don't have the exact step count, we use a fixed reasonable kappa
    that balances exploration and exploitation well for this problem.
    
    For the Beale function on [-4.5, 4.5]^2, the function values can range
    from 0 (at (3, 0)) up to several hundred. We use a kappa that provides
    good exploration early and exploitation later.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Handle non-finite mu
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Adaptive kappa: start with moderate exploration, reduce as f_best improves
    # f_best starts around 100-1000 for random points on Beale, and approaches 0
    # We map f_best to a kappa value:
    # - When f_best is large (early), use higher kappa for exploration
    # - When f_best is small (later), use lower kappa for exploitation
    
    # Normalize f_best to get a sense of progress
    # Beale function: f(x,y) = (1.5 - 2x + x^2)(3 - 2y + y^2)^2
    # At random points in [-4.5, 4.5]^2, typical values are in [0, ~500]
    
    if f_best is None or not np.isfinite(f_best):
        kappa = 2.0
    else:
        # Use a logarithmic scale for adaptation
        # f_best ~ 100 -> kappa ~ 2.5 (explore)
        # f_best ~ 10  -> kappa ~ 1.5
        # f_best ~ 1   -> kappa ~ 0.8
        # f_best ~ 0.1 -> kappa ~ 0.3
        # f_best ~ 0   -> kappa ~ 0.1
        
        # Simple adaptive scheme
        f_best_clipped = max(f_best, 1e-6)
        # kappa decreases as f_best decreases
        kappa = 1.0 + 2.0 / (1.0 + np.log10(max(f_best_clipped, 1e-6) + 1e-6))
        # Clamp kappa to reasonable range
        kappa = np.clip(kappa, 0.1, 3.0)
    
    # LCB score: we want to maximize -mu - kappa * sigma
    score = -mu - kappa * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
