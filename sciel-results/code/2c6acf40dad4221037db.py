"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining probability of improvement
    with an uncertainty (exploration) term, normalized by cost.
    
    Uses a robust, bounded signal that balances exploitation and exploration
    in a short budget setting.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement (bounded in [0, 1])
    p_imp = norm.cdf(z)
    
    # Expected improvement (for exploitation)
    ei = (f_best - mu) * p_imp + sigma_safe * norm.pdf(z)
    
    # Exploration term: reward high uncertainty
    # Use a bounded transformation to keep it comparable to EI
    exploration = sigma_safe * (1.0 - p_imp)  # high when sigma is large and improvement unlikely
    
    # Combine: weight exploitation and exploration
    # In a short budget, we want to be more aggressive with exploitation
    # once we have some information, but explore early
    # Use a simple blend that favors exploitation as f_best decreases
    blend = 0.7 * ei + 0.3 * exploration
    
    # Normalize by cost
    scores = blend / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
