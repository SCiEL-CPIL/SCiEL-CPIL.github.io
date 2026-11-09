"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition using multiplicative gate based on signal-to-noise ratio
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative (EI should be >= 0, but numerical issues can cause negatives)
    ei = np.maximum(ei, 0.0)
    
    # Calibrate k based on the scale of EI and cost
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    # We want to suppress high-cost points unless EI is large
    # Use a moderate k value that balances exploration and exploitation
    k = 1.5
    
    # Multiplicative gate: exp(-k * (cost - 1))
    # When cost=1 (cheapest), gate=1; when cost=2 (most expensive), gate=exp(-k)
    gate = np.exp(-k * (cost - 1.0))
    
    # Final score: EI multiplied by the cost gate
    score = ei * gate
    
    # Add a small exploration bonus for points with high uncertainty
    # This helps escape local optima in the acquisition landscape
    explore_weight = 0.05
    explore_term = explore_weight * sigma_safe * gate
    
    score = score + explore_term
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
