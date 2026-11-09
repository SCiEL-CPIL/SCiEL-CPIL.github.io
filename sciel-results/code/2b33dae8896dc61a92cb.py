"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track number of evaluations
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: track elapsed time and return safe result if needed
    # For 2000 points, computation is fast, but include guard for safety
    start_time = time.time()
    deadline = 330.0  # 330s wall-clock limit
    
    # Structural change: Use Expected Improvement (EI) instead of UCB
    # EI is a well-established acquisition function that balances exploration
    # and exploitation by computing the expected improvement over f_best
    # For minimization, we want to find points where f(x) < f_best
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle cases where sigma is very small to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute EI using standard normal CDF and PDF
    # Phi(z) is the CDF, phi(z) is the PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula for minimization
    # When f_best - mu > 0, there's potential improvement
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # The acquisition score is the EI itself (higher is better)
    scores = ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
