"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 280.0
_start_time = time.time()


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid acquisition: blends Quantile of the Posterior (QoP)
    with a normalized uncertainty bonus, both divided by cost^gamma.
    
    The QoP term targets the p-th quantile of the posterior (optimistic for
    minimization). The uncertainty bonus encourages exploration of regions
    where the GP is uncertain, helping escape local minima in the multimodal
    rastrigin landscape. The blend weight adapts to f_best: early on (high
    f_best) we weight exploration more; later (low f_best) we weight
    exploitation more.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Fixed quantile p=0.25 (measured better than adaptive in prior experiments)
    p = 0.25
    z_p = norm.ppf(p)  # ≈ -0.6745
    
    # QoP: p-th quantile of the posterior (optimistic for minimization)
    qop = mu + sigma_safe * z_p
    
    # Normalized uncertainty: scale sigma to [0, 1] across candidates
    sigma_min = sigma_safe.min()
    sigma_max = sigma_safe.max()
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Adaptive blend weight: early (high f_best) -> more exploration;
    # late (low f_best) -> more exploitation.
    # f_best for rastrigin 2d on [-5.12, 5.12]^2 ranges from ~20 (initial) to 0 (optimal).
    # progress = 1 when f_best is good (low), 0 when f_best is bad (high).
    progress = 1.0 - np.clip(f_best / 20.0, 0.0, 1.0)
    
    # Blend weight: exploration_weight decreases from 0.3 to 0.05 as progress increases
    exploration_weight = 0.3 - 0.25 * progress
    
    # Scale the uncertainty bonus to be comparable to the QoP term magnitude
    # Use the typical magnitude of -qop as a reference scale
    qop_magnitude = np.abs(qop).max()
    if qop_magnitude < 1e-12:
        qop_magnitude = 1.0
    
    # Combined objective value (to be minimized)
    combined = qop - exploration_weight * qop_magnitude * sigma_norm
    
    # Cost-aware: divide by cost^gamma with gamma=1.3 for stronger cost penalty
    gamma = 1.3
    cost_penalty = cost_safe ** gamma
    
    # Score: higher is better, so negate the combined objective
    scores = -combined / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
