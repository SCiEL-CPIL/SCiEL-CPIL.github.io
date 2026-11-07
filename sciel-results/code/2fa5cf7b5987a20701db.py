"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) with a
    lower-confidence-bound (LCB) / Thompson-sampling-like exploration term.
    
    We use a weighted sum that emphasizes exploitation when sigma is small
    and exploration when sigma is large, but with a different functional form
    than pure EI to avoid the known pitfalls of EI in low-dimensional,
    multi-modal functions like Michalewicz 2D.
    
    Strategy: Use a modified EI that incorporates a penalty for regions
    where sigma is very small (to avoid over-exploiting already-well-known
    regions) and a bonus for regions with moderate uncertainty.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Standard EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower confidence bound (LCB) for minimization:
    # We want low mu and high sigma (uncertainty)
    # LCB score = -mu - beta * sigma  (higher is better for selection)
    # But we need to convert to a "higher is better" score comparable to EI
    
    # A more robust approach: use a combination that balances
    # exploitation (low mu) and exploration (high sigma)
    
    # Normalize sigma to [0, 1] range for the current candidate set
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-10
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Normalize mu relative to f_best
    mu_range = f_best - np.min(mu) + 1e-10
    mu_norm = np.clip((f_best - mu) / mu_range, 0, 1)
    
    # Weighted combination: 
    # - Exploitation: prefer low mu (high mu_norm)
    # - Exploration: prefer high sigma (high sigma_norm)
    # Use a non-linear weighting to create a more aggressive exploration profile
    
    # The key insight: Michalewicz 2D has a global minimum that is hard to find
    # because it's in a narrow valley. Pure EI tends to get stuck in local minima.
    # We need more exploration early on.
    
    # Use a temperature-like parameter that increases exploration
    # The exponent on sigma_norm controls how strongly we weight uncertainty
    exploration_weight = 0.5
    exploitation_weight = 0.5
    
    # Score: higher means more attractive
    # Use a power-law to emphasize exploration in uncertain regions
    score = exploitation_weight * mu_norm + exploration_weight * (sigma_norm ** 0.5)
    
    # Add a small EI component to ensure we still exploit promising regions
    # Scale EI to be comparable to the [0,1] range
    ei_max = np.max(ei) + 1e-10
    ei_norm = ei / ei_max
    
    # Final score: blend EI with the exploration/exploitation balance
    final_score = 0.3 * ei_norm + 0.7 * score
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
