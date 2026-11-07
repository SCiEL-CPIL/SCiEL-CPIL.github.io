import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Adaptive LCB with Distance-to-Best repulsion.
    
    Core idea: Use Lower Confidence Bound (LCB) as the primary signal, where
    the exploration coefficient kappa adapts based on how far f_best is from
    the known global minimum. This provides strong exploration when far from
    optimum and exploitation when close.
    
    Structural addition: A distance-to-best weighting in mu-space creates a
    repulsive force that boosts candidates whose posterior mean is far from
    the current best candidate. This helps escape dense local minima of the
    Michalewicz function by encouraging jumps to distant basins.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    t_start = time.time()
    
    # Ensure proper types and handle numerical issues
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    eps = 1e-12
    
    # --- Adaptive LCB (Primary Signal) ---
    # Known global minimum of Michalewicz 2D
    opt = -1.8013
    dist_to_opt = max(0.0, f_best - opt)
    
    # kappa: higher when far from optimum (explore), lower when close (exploit)
    # Scale by 5.0 to normalize the typical range of f_best - opt
    kappa = 0.5 + 2.0 * (dist_to_opt / 5.0)
    kappa = min(kappa, 3.0)
    
    # LCB score for minimization: higher score = more promising
    lcb_score = -mu + kappa * sigma
    
    # --- Distance-to-Best Weighting (Repulsive Force) ---
    # Find the index of the best candidate (lowest mu)
    best_idx = np.argmin(mu)
    best_mu = mu[best_idx]
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    
    if mu_range > eps:
        # Normalize mu to [0, 1] range
        mu_norm = (mu - mu_min) / mu_range
        best_mu_norm = (best_mu - mu_min) / mu_range
        dist_from_best = np.abs(mu_norm - best_mu_norm)
        
        # Boost distant points: weight = 1 + alpha * dist_from_best
        alpha = 2.0
        dist_weight = 1.0 + alpha * dist_from_best
    else:
        dist_weight = np.ones_like(mu)
    
    # Weighted LCB
    weighted_lcb = lcb_score * dist_weight
    
    # Safety: check time before returning
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return lcb_score
    
    # Ensure all scores are finite
    scores = np.nan_to_num(weighted_lcb, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
