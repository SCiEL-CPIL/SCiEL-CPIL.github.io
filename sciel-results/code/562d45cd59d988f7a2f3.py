"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of Expected Improvement
    and Upper Confidence Bound with adaptive cost-awareness.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
    2. Compute UCB-style exploration term with adaptive beta.
    3. Combine EI and exploration with a cost-aware penalty that strengthens
       as the budget depletes (proxied by f_best approaching 0).
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(2000, dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    Z = (f_best - mu) / sigma_safe
    Z = np.clip(Z, -10.0, 10.0)
    
    from scipy.stats import norm
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # UCB exploration term: -mu + beta * sigma
    # Beta adapts based on how close we are to the optimum
    f_best_abs = abs(f_best)
    # When f_best is large (far from optimum), explore more aggressively
    # When f_best is small (near optimum), exploit more
    beta = 0.5 + 1.5 / (1.0 + f_best_abs)
    beta = np.clip(beta, 0.3, 3.0)
    
    # For minimization, UCB-like score: lower mu is better, higher sigma is good for exploration
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both EI and UCB to comparable scales
    ei_max = np.max(EI)
    if not np.isfinite(ei_max) or ei_max <= 0:
        ei_max = 1.0
    
    ucb_max = np.max(ucb_score)
    ucb_min = np.min(ucb_score)
    ucb_range = ucb_max - ucb_min
    if not np.isfinite(ucb_range) or ucb_range <= 0:
        ucb_range = 1.0
    
    # Blend EI and UCB
    # Weight shifts from exploration to exploitation as f_best decreases
    w_ei = 0.6 + 0.3 / (1.0 + f_best_abs)
    w_ei = np.clip(w_ei, 0.5, 0.9)
    w_ucb = 1.0 - w_ei
    
    EI_norm = EI / ei_max
    UCB_norm = (ucb_score - ucb_min) / ucb_range
    
    combined = w_ei * EI_norm + w_ucb * UCB_norm
    
    # Cost-aware penalty: prefer cheaper evaluations
    # Cost is in [1, 2], so (cost - 1) is in [0, 1]
    # Alpha increases as we get closer to optimum to conserve budget
    alpha = 0.3 + 0.7 / (1.0 + f_best_abs)
    alpha = np.clip(alpha, 0.2, 1.5)
    
    cost_penalty = 1.0 + alpha * (np.maximum(cost, 1.0) - 1.0)
    score = combined / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
