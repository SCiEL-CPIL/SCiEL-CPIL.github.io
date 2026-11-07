import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UWI acquisition for Styblinski-Tang 2D BO.
    Uses Uncertainty-Weighted Improvement with local baseline estimation.
    """
    t0 = time.time()
    
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Progress measure based on f_best
    known_min = -78.3323
    progress = np.clip(1.0 - (f_best - known_min) / 200.0, 0.0, 1.0)
    
    # Adaptive beta: start high (explore), decrease over time
    beta = 3.0 - 2.5 * progress
    
    # Local baseline: use a soft-min of mu values to estimate local context
    # This captures the "explored basin" level rather than just global best
    # Use log-sum-exp for numerical stability
    mu_min = np.min(mu)
    # Soft-min with temperature that adapts to progress
    # Early: wide temperature (more global), Late: narrow (more local)
    temp = 5.0 * (1.0 - progress) + 0.5
    mu_scaled = mu / temp
    mu_max = np.max(mu_scaled)
    log_sum_exp = mu_max + np.log(np.sum(np.exp(mu_scaled - mu_max)))
    local_baseline = temp * log_sum_exp  # This is actually a soft-max, we want soft-min
    
    # For minimization, soft-min = -soft_max(-mu)
    neg_mu_scaled = -mu / temp
    neg_mu_max = np.max(neg_mu_scaled)
    log_sum_exp_neg = neg_mu_max + np.log(np.sum(np.exp(neg_mu_scaled - neg_mu_max)))
    local_baseline = -temp * log_sum_exp_neg
    
    # Uncertainty-Weighted Improvement (UWI):
    # Improvement over local baseline, weighted by relative uncertainty
    # UWI = (local_baseline - mu) * (sigma / (sigma + local_sigma_ref))
    # where local_sigma_ref is a reference scale for uncertainty
    
    # Reference sigma scale: median of sigma values
    sigma_ref = np.median(sigma)
    
    # Relative uncertainty weight: high where sigma is large relative to reference
    # This targets boundaries of explored basins
    rel_uncertainty = sigma / (sigma + sigma_ref)
    
    # Improvement: how much better than local baseline
    improvement = local_baseline - mu
    
    # UWI score: improvement weighted by relative uncertainty
    # Plus a UCB component for exploration
    uwi = improvement * rel_uncertainty + beta * sigma
    
    # Cost-aware: divide by cost
    score = uwi / (cost ** 1.2)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Final safety check on time
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    return score
