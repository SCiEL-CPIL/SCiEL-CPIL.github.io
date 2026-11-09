"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) for minimization.
    
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    We apply a penalty on the uncertainty term to prevent high-noise regions
    from dominating in the sparse data regime. The penalty scales the sigma
    term by a factor that decreases as sigma increases relative to the
    improvement potential, effectively requiring higher confidence for
    uncertain points.
    
    The final score is EI / cost to account for evaluation cost.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Penalty on uncertainty: scale the sigma*phi_z term down
    # When sigma is large relative to the improvement (f_best - mu),
    # we want to reduce the exploration incentive.
    # Use a soft penalty: multiply the uncertainty term by exp(-alpha * sigma / max(f_best - mu, eps))
    # But this can be tricky. Instead, use a simpler approach:
    # Blend between pure exploitation (mu-based) and exploration (sigma-based)
    # based on confidence.
    
    # Alternative: use a weighted combination where the weight on exploration
    # decreases as sigma increases
    # Let's use: score = (f_best - mu) * Phi_z + beta * sigma * phi_z
    # where beta < 1 to reduce exploration
    # But we want it adaptive. Let's use beta = 1 / (1 + sigma / scale)
    
    # Scale parameter: how much sigma we consider "high"
    # Since f_best is the best observed, and we're in [0,1]^2 scaled,
    # typical improvements might be O(0.1-1.0)
    scale = 0.1
    beta = 1.0 / (1.0 + sigma / scale)
    
    # Recompute EI with penalized uncertainty
    ei_penalized = (f_best - mu) * Phi_z + beta * sigma * phi_z
    
    # Ensure non-negative (EI can be slightly negative due to numerical issues)
    ei_penalized = np.maximum(ei_penalized, 0.0)
    
    # Cost-aware: divide by cost
    scores = ei_penalized / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
