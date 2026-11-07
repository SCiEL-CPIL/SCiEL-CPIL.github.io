# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses a hybrid of Expected Improvement (EI) and Probability of Improvement (PI).
    The blend provides a balance between exploitation (large expected gain) and
    exploration (high probability of any gain), which can help escape local optima
    or narrow regions where EI might be too conservative.
    """
    start_time = time.time()
    deadline = 300.0  # 300s limit, keep margin for safety

    # Convert to arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Sanitize inputs
    # Ensure sigma is strictly positive to prevent division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Replace non-finite mu with f_best (worst case assumption for unknowns)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Z-score
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip Z to prevent extreme values in exp/cdf
    z = np.clip(z, -10.0, 10.0)
    
    # Compute EI
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    phi_z = norm.pdf(z)
    Phi_Z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_z
    
    # Compute PI
    # PI = Phi(Z)
    pi = Phi_Z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Hybrid score: EI + PI
    # This adds a constant exploration bias from PI to the exploitation signal of EI.
    # PI is bounded [0, 1], while EI can be larger.
    # The addition encourages points that have a reasonable chance of improvement
    # even if the expected magnitude is small, potentially helping in flat regions.
    score = ei + pi
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return zeros to ensure validity, though EI is very fast
        return np.zeros_like(score)
        
    return score
# EVOLVE-BLOCK-END
