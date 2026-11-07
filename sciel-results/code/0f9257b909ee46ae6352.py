# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf, erfc

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a numerically stable Expected Improvement
    with a bounded exploration bonus for robustness in low-noise GPs.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -100, 100)
    
    # Stable computation of Phi(z) and phi(z)
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    
    sqrt2 = np.sqrt(2.0)
    
    # Phi(z) using erfc for better stability
    Phi = 0.5 * erfc(-z / sqrt2)
    
    # phi(z) with underflow protection
    log_phi = -0.5 * z * z - 0.5 * np.log(2.0 * np.pi)
    # Clip log_phi to avoid underflow to 0 for very negative values
    log_phi = np.maximum(log_phi, -700.0)  # exp(-700) ~ 1e-304, near min float
    phi = np.exp(log_phi)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # But when sigma is very small and mu < f_best, EI should be ~ f_best - mu
    # When sigma is very small and mu > f_best, EI should be ~ 0
    # The formula above handles this, but let's ensure numerical stability
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Add a small bounded exploration bonus that only matters when sigma is significant
    # This helps escape local optima without dominating when GP is confident
    # The bonus is proportional to sigma but capped to prevent it from overriding strong exploitation signals
    exploration_bonus = 0.1 * sigma
    
    # Combine: the exploration bonus should not exceed a fraction of the typical EI scale
    # Scale the bonus relative to f_best magnitude to keep it dimensionally consistent
    scale = max(abs(f_best), 1.0)
    exploration_bonus = np.minimum(exploration_bonus, 0.5 * scale)
    
    score = ei + exploration_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
