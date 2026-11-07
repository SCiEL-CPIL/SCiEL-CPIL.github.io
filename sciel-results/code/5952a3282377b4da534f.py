"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Hartmann 6d).
    Combines Expected Improvement (EI) with Upper Confidence Bound (UCB)
    using an adaptive blend, normalized by cost.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Deadline guard: if we're running too long, return a simple fallback
    if time.time() - t_start > 280.0:
        kappa = 1.0
        score = (f_best - mu + kappa * sigma_safe) / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization: f_best - (mu - kappa * sigma)
    # Higher kappa means more exploration
    kappa = 1.2
    ucb = f_best - mu + kappa * sigma_safe
    
    # Adaptive blending: when uncertainty is high relative to potential improvement,
    # lean more toward UCB; when confident, lean toward EI
    # uncertainty_ratio is high when sigma is large compared to |f_best - mu|
    uncertainty_ratio = sigma_safe / (np.abs(f_best - mu) + sigma_safe + 1e-9)
    blend = np.clip(uncertainty_ratio, 0.0, 1.0)
    
    # Smooth the blend: moderate weight on UCB
    blend = 0.5 * blend
    
    # Combine EI and UCB
    combined = (1.0 - blend) * ei + blend * ucb
    
    # Normalize by cost: higher cost reduces the score
    scores = combined / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
