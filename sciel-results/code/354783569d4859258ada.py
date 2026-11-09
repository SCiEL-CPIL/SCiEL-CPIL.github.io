import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural blend:
    - Lower-tail expected improvement (for minimization)
    - Uncertainty-weighted exploration that decays with confidence
    - Cost normalization to prefer cheap evaluations
    """
    t_start = time.time()
    
    # Safety for sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Z-score for improvement (lower is better, so we want mu < f_best)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = E[min(0, f(x) - f_best)] = (mu - f_best) * Phi(-z) + sigma * phi(z)
    # But since we want to maximize acquisition (higher = better), we flip:
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize EI and PI to [0, 1]
    ei_min, ei_max = np.min(ei), np.max(ei)
    pi_min, pi_max = np.min(pi), np.max(pi)
    
    ei_range = ei_max - ei_min + 1e-9
    pi_range = pi_max - pi_min + 1e-9
    
    ei_norm = (ei - ei_min) / ei_range
    pi_norm = (pi - pi_min) / pi_range
    
    # Uncertainty exploration term: relative sigma
    max_sigma = np.max(sigma_safe) + 1e-9
    sigma_rel = sigma_safe / max_sigma
    
    # Structural blend:
    # - 0.4 * EI (magnitude of expected improvement)
    # - 0.3 * PI (probability of improvement) 
    # - 0.3 * exploration (uncertainty bonus, decays as GP becomes confident)
    blended = 0.4 * ei_norm + 0.3 * pi_norm + 0.3 * sigma_rel
    
    # Cost-aware: divide by cost to prefer cheap evaluations
    cost_safe = np.maximum(cost, 1e-10)
    scores = blended / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard: if we're taking too long, return immediately
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
