"""Rewritten acquisition for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: total wall-clock budget for the whole evaluator run is 330s.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    LCB = mu - kappa * sigma
    
    We use an adaptive kappa that starts high (exploration) and decreases
    as we get closer to the optimum (exploitation).
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Return a deterministic fallback: prefer low mu
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr.copy()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known minimum of Styblinski-Tang 1D
    f_min = -39.1662
    
    # Adaptive kappa:
    # When f_best is far from f_min (early), use large kappa for exploration
    # When f_best is close to f_min (late), use small kappa for exploitation
    
    # Progress metric: how much improvement is still possible
    # Typical initial f_best ~ 20-50, final ~ -39
    # Define a smooth transition
    
    # Use a sigmoid based on f_best
    # center: midpoint between typical initial and final
    # width: controls transition speed
    
    center = -5.0
    width = 15.0
    
    # t ~ 1 when f_best >> center (early), t ~ 0 when f_best << center (late)
    t = 1.0 / (1.0 + np.exp(-(f_best - center) / width))
    
    # kappa ranges from kappa_max (early) to kappa_min (late)
    kappa_max = 2.5
    kappa_min = 0.3
    
    kappa = kappa_min + t * (kappa_max - kappa_min)
    
    # LCB score: we want to minimize mu - kappa*sigma
    # But acquisition returns higher = evaluate next
    # So we negate: score = -(mu - kappa*sigma) = -mu + kappa*sigma
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
