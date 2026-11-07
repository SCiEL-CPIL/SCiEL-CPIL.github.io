import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TOTAL_BUDGET = 30.0
_INITIAL_COST = 5 * 1.5  # 5 initial points, mean cost ~1.5

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return 1.0 / cost

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    # Expected Improvement (numerically stable)
    # For z > 0: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For z <= 0: EI = (f_best - mu) * Phi(z) + sigma * phi(z) (same formula)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    cost_safe = np.maximum(cost, eps)
    ei_cost = ei / cost_safe
    
    # Add exploration bonus weighted by remaining budget fraction
    # Early on: more exploration; later: more exploitation
    total_cost_so_far = np.mean(cost_safe)  # approximate
    remaining_fraction = max(0.0, 1.0 - total_cost_so_far / _TOTAL_BUDGET)
    
    # UCB-like exploration term
    ucb = 0.5 * sigma_safe / cost_safe
    
    # Blend: early search uses more exploration, later uses more exploitation
    blend = 0.3 * remaining_fraction
    score = (1.0 - blend) * ei_cost + blend * ucb
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
