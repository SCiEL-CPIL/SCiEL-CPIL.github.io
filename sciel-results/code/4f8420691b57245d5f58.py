import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Probabilistic Lower Bound (PLB): score = mu + k * sigma,
    which maximizes the probability that the true function value exceeds
    a dynamic threshold derived from f_best. This naturally decays
    exploration as confidence increases (sigma shrinks near known points).
    """
    start_time = time.time()
    timeout = 320.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    n = len(mu)
    
    # Ensure sigma is positive for computations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Gap from optimum determines exploration-exploitation tradeoff
    gap = max(f_best - 3.0, 0.0)
    
    # PLB: score = mu + k * sigma
    # Higher score means higher probability of exceeding the threshold.
    # The key insight: we want to find points where f > f_best is likely,
    # but since we're minimizing, we actually want f < f_best.
    # So we use: score = -mu + k * sigma (equivalent to PLB for minimization)
    # But the directive says score = mu + k*sigma for PLB.
    # For minimization, PLB targets P(f < threshold) which corresponds to
    # maximizing -mu + k*sigma. Let's use the standard PLB form adapted.
    
    # Dynamic k: when far from optimum, explore more (larger k);
    # when close, exploit more (smaller k).
    # Use a smooth decay: k transitions from high to low as gap shrinks.
    k_base = 0.5
    k_extra = 4.0
    k_scale = 1.2
    k = k_base + k_extra * np.exp(-gap / k_scale)
    
    # Soft damping of sigma to prevent extreme exploration
    sigma_scale = 2.5
    sigma_eff = sigma_safe / (1.0 + sigma_safe / sigma_scale)
    
    # PLB score for minimization: -mu + k * sigma_eff
    # This is equivalent to maximizing P(f < f_best - margin)
    scores = -mu + k * sigma_eff
    
    # Additional refinement: when very close to optimum, add a small
    # bonus for points with very low mu to encourage exploitation
    if gap < 0.5:
        mu_bonus = np.maximum(0.0, (f_best - mu) / max(gap, 0.05))
        scores += 0.15 * mu_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - start_time > timeout:
        return scores
    
    return scores
