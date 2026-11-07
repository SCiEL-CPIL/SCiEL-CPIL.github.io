"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # Clip EI to be bounded and avoid extreme values
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    # Clip EI to [0, 0.05] to keep it bounded and comparable to PoI scale
    ei_clipped = np.clip(ei, 0.0, 0.05)
    
    # Combine: primary PoI with small EI contribution for exploitation
    # beta=0.5 means EI contributes up to 0.025 (0.5 * 0.05) to the score
    # PoI is in [0,1], so EI is a modest boost
    beta = 0.5
    combined = poi + beta * ei_clipped
    
    # Cost-awareness: use slightly super-linear penalty
    # cost is in [1, 2], cost^1.1 gives mild additional penalty for expensive points
    cost_penalty = np.power(cost, 1.1)
    
    scores = combined / cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
