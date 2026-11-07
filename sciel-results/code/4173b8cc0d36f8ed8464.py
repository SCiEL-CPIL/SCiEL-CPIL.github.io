"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: return a simple, fast result if we are close to the 330s limit
    # 300s provides a 30s safety margin.
    if time.time() - _start_time > 300:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return poi / cost_safe

    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (POI)
    # POI = P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the acquisition, so we use the negative of the lower confidence bound.
    # LCB = mu - beta * sigma. Maximize -LCB = beta * sigma - mu.
    # beta controls exploration. 2.0 is a standard choice.
    beta = 2.0
    ucb = beta * sigma_safe - mu
    
    # Blend EI, POI, and UCB
    # Based on inspirations:
    # - Inspiration 1 (score 1.2143) used POI + 0.5 * UCB_scaled
    # - Current (score 1.1624) used 0.4 * EI + 0.6 * UCB
    # - Inspiration 2 (score 1.1749) used 0.6 * UCB + 0.4 * EI with log-normal cost
    
    # We combine the structural idea of POI+UCB (Inspiration 1) with the refinement of 
    # scaling UCB to be comparable to POI (both in [0,1] range) and adding a small EI component.
    
    # Scale UCB to be comparable to POI (which is in [0, 1])
    # Use the magnitude of f_best as a natural scale, with a floor to avoid division by small numbers
    scale = np.abs(f_best) + 1.0
    ucb_scaled = np.maximum(ucb, 0.0) / scale
    
    # Combine: POI is primary (exploitation), UCB is secondary (exploration), EI provides probabilistic dampening
    # Weighting: 1.0 * POI + 0.5 * UCB_scaled + 0.2 * EI/scale
    # The EI term is scaled to be comparable to POI and UCB
    ei_scaled = np.maximum(ei, 0.0) / scale
    
    blended_score = poi + 0.5 * ucb_scaled + 0.2 * ei_scaled
    
    # Cost-aware adjustment
    # Higher score means more likely to be evaluated.
    # We divide by cost to prefer cheaper evaluations.
    # Cost is in [1, 2], so this is a mild penalty.
    # Use log-normal cost model for slightly stronger penalty on expensive points
    # (from Inspiration 2)
    cost_mean = cost_safe
    cost_std = 0.1 * cost_mean
    
    var = cost_std ** 2
    sigma_c_sq = np.log(1.0 + var / (cost_mean ** 2))
    sigma_c = np.sqrt(sigma_c_sq)
    mu_c = np.log(cost_mean) - 0.5 * sigma_c_sq
    
    # E[C^{-1}] = exp(-mu_c + 0.5 * sigma_c^2)
    e_inv_c = np.exp(-mu_c + 0.5 * sigma_c_sq)
    
    final_score = blended_score * e_inv_c
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
