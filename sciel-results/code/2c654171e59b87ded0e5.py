# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: EI with adaptive exploration bonus.
    """
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: clamp sigma and cost to positive values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Adaptive exploration bonus:
    # In early iterations (when f_best is still high / uncertainty is large relative to improvement),
    # add a small bonus proportional to sigma to encourage exploring uncertain regions.
    # The bonus is scaled so it doesn't overwhelm EI.
    # Use a soft gate: bonus is active when mean sigma is large relative to typical improvement.
    mean_sigma = np.mean(sigma_safe)
    mean_improvement = max(f_best - np.mean(mu), 1e-9)
    # ratio > 1 means uncertainty dominates -> more exploration
    ratio = mean_sigma / mean_improvement
    # Gate: 0 when ratio < 0.5, ramps to 1 when ratio > 2
    gate = np.clip((ratio - 0.5) / 1.5, 0.0, 1.0)
    
    # Exploration bonus: small fraction of sigma, gated
    exploration_bonus = gate * 0.1 * sigma_safe
    
    # Combine: EI dominates, small exploration bonus
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
