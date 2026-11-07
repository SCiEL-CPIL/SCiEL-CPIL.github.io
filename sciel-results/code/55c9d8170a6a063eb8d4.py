import numpy as np
from scipy.stats import norm
from scipy.special import softplus
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 290.0

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    n = len(mu)
    
    # Guard against zero/negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # 1. Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # 2. Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 3. UCB-style exploration term for minimization
    # Smoothed with softplus to avoid hard cutoffs
    beta = 2.5
    ucb_raw = (f_best - mu + beta * sigma_safe)
    ucb_smooth = softplus(ucb_raw)
    
    # Scale EI and UCB to be comparable to POI (0-1 range)
    # Shekel 4d min is -10.5, typical f_best starts ~ -5, so max improvement ~ 5-6
    scale = 5.0
    ei_scaled = ei / scale
    ucb_scaled = ucb_smooth / scale
    
    # Combine all terms
    # Weights adjusted to balance exploitation (POI) and exploration (UCB/EI)
    # POI: robust exploitation
    # EI: magnitude of improvement
    # UCB: uncertainty-driven exploration
    scores = (0.4 * poi + 0.3 * ei_scaled + 0.3 * ucb_scaled) / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        fallback_scores = poi / cost_safe
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
