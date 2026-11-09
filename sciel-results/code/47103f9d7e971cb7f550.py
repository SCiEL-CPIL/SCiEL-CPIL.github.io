# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).
    Uses Cost-Adjusted Probability of Improvement (PoI) weighted by inverse cost,
    plus a Gaussian Process uncertainty bonus.
    
    score(x) = P(mu < f_best) / cost(x) + beta * sigma(x) / cost(x)
    
    This explicitly prioritizes candidates with high likelihood of success per unit
    budget spent, rather than maximizing expected magnitude of improvement.
    """
    t_start = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.size

    # Numerical guards
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Probability of Improvement: P(y < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -40.0, 40.0)
    p_improve = norm.cdf(z)
    
    # Cost-adjusted PoI: prioritize high probability of improvement per unit cost
    cost_adjusted_poi = p_improve / cost_safe
    
    # Uncertainty bonus: encourage exploration in uncertain regions, also cost-adjusted
    # Use a small coefficient to balance exploration
    beta = 0.1
    uncertainty_bonus = beta * sigma_safe / cost_safe
    
    # Combine: primary signal is cost-adjusted PoI, secondary is cost-adjusted uncertainty
    final_scores = cost_adjusted_poi + uncertainty_bonus

    # Ensure all values are finite (higher = evaluate next)
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)

    # Deadline guard (safety margin before the 330 s kill)
    if (time.time() - t_start) > 300.0:
        return final_scores

    return final_scores
# EVOLVE-BLOCK-END
