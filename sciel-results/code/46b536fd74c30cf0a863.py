"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) per unit cost.
    
    Strategy:
    1. We want to maximize the expected reduction in simple regret per unit cost.
    2. Standard EI for minimization is:
       EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
       where z = (f_best - mu(x)) / sigma(x)
       Phi is the standard normal CDF and phi is the standard normal PDF.
    3. The cost-aware score is EI(x) / cost(x).
    
    This differs structurally from the previous LCB/cost approach:
    - LCB is linear in sigma and mu.
    - EI is non-linear, involving CDF and PDF, which naturally balances exploration 
      (high sigma) and exploitation (low mu relative to f_best) more optimally 
      for the specific goal of minimizing simple regret.
    - Dividing by cost ensures that expensive evaluations (higher cost) are penalized 
      proportionally to their expected gain.
    
    Implementation:
    - Handle sigma == 0 cases to avoid division by zero in z.
    - Ensure outputs are finite.
    """
    # Ensure sigma is non-negative and handle zeros
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI components
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative. Phi(z) is small, but (f_best-mu) is negative.
    # The term (f_best - mu) * Phi(z) will be negative (bad).
    # The term sigma * phi(z) is positive.
    # Usually EI is defined as max(0, ...), but for acquisition functions, 
    # we often use the raw expectation which can be negative, or clip at 0.
    # Clipping at 0 is standard to avoid selecting points with negative expected improvement.
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # cost is in [1, 2], so it's safe to divide.
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
