"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware EI with uncertainty-weighted exploration.
    # The standard EI term captures exploitation.
    # We add a small exploration bonus proportional to sigma to encourage
    # visiting high-uncertainty regions, which helps escape local minima
    # in the Levy function's rugged landscape.
    # The exploration weight is scaled down as f_best approaches 0 (the true minimum),
    # since once we're close to the global minimum, exploitation should dominate.
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourages visiting uncertain regions
    # Scale down as we get closer to the known minimum
    # When f_best is close to 0, exploration weight decreases
    # When f_best is large, exploration weight is higher
    exploration_weight = np.minimum(f_best / (f_best + 1.0), 1.0)
    exploration_bonus = exploration_weight * 0.1 * sigma_safe
    
    # Combine EI and exploration, then normalize by cost
    score = (ei + exploration_bonus) / cost
    
    # Ensure finite values
    return np.where(np.isfinite(score), score, 0.0)
# EVOLVE-BLOCK-END
