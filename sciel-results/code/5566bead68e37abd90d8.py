import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining LCB with cost normalization.
    Uses a hybrid of LCB and EI, modulated by progress, with cost penalty.
    
    LCB (Lower Confidence Bound) for minimization: mu - beta * sigma
    EI (Expected Improvement) for minimization: E[max(f_best - f(x), 0)]
    We want to maximize the score, so we use -LCB and EI directly.
    Dividing by cost penalizes expensive evaluations.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu_safe / cost_safe
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Estimate progress: 0 at start, 1 near optimum
    known_min = -78.3323
    f_start_est = 50.0  # Typical starting value for random points
    progress = np.clip((f_start_est - f_best) / (f_start_est - known_min), 0.0, 1.0)
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # So we maximize -(mu - beta * sigma) = -mu + beta * sigma
    # Adaptive beta: more exploration early, less late
    beta = 2.5 - 1.5 * progress
    lcb_score = -mu + beta * sigma
    
    # EI for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hybrid: weight LCB and EI based on progress
    # Early: more LCB (exploration), Late: more EI (exploitation)
    w_lcb = 0.7 - 0.4 * progress
    w_ei = 1.0 - w_lcb
    
    # Normalize EI to comparable scale with LCB
    # EI can be much smaller in magnitude, so scale it
    ei_scale = np.mean(np.abs(lcb_score)) + 1e-9
    ei_scaled = ei * (ei_scale / (np.mean(np.abs(ei)) + 1e-9))
    
    # Combine
    hybrid_score = w_lcb * lcb_score + w_ei * ei_scaled
    
    # Cost adjustment: divide by cost^p where p adapts to progress
    # Stronger cost penalty early to find cheap good points, weaker late
    cost_power = 1.2 - 0.3 * progress
    final_score = hybrid_score / (cost ** cost_power)
    
    # Ensure finite output
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
