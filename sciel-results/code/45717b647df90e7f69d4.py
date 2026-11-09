"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function for minimizing Griewank 2D.
    
    EI provides a principled balance between exploration and exploitation by computing the
    expected improvement over the current best. The uncertainty term is naturally damped
    by the standard deviation, preventing domination by high-noise regions in sparse data regimes.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best (lowest) observed value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # f_best is the best observed value (lowest since we minimize)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0
    
    # Compute Z-score: z = (f_best - mu) / sigma
    # Positive z means mu < f_best (potential improvement)
    z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in CDF/PDF evaluation
    z = np.clip(z, -10, 10)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Normalizing by cost ensures expensive points are penalized appropriately
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
