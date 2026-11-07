# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (cEI) acquisition function.
    
    This approach uses Expected Improvement (EI) to balance exploration and exploitation.
    EI estimates the expected amount by which a point improves upon the current best.
    To account for the cost of evaluation, we divide the EI by the cost (cEI).
    
    Steps:
    1. Calculate the z-score (beta) for each candidate: beta = (f_best - mu) / sigma.
    2. Calculate the Expected Improvement: EI = (f_best - mu) * Phi(beta) + sigma * phi(beta).
       Note: For minimization, improvement is f_best - mu.
       The standard EI formula for maximization is (mu - f_best) * Phi(-beta) + sigma * phi(beta).
       For minimization, let improvement = f_best - f(x). 
       E[improvement] = E[max(0, f_best - f(x))]
       Let Z ~ N(0,1). f(x) = mu + sigma*Z.
       f_best - f(x) = (f_best - mu) - sigma*Z.
       Let A = f_best - mu.
       E[max(0, A - sigma*Z)] = A * P(A - sigma*Z > 0) + E[(A - sigma*Z) * 1(A - sigma*Z > 0)]
       This simplifies to A * Phi(A/sigma) + sigma * phi(A/sigma) if A > 0? 
       Actually, standard result for minimization:
       EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
       This is valid when f_best > mu. If f_best < mu, EI is 0.
       The formula above handles this naturally if we ensure terms are non-negative or use max.
       However, phi is always positive. Phi(x) is close to 0 for large negative x.
       If f_best < mu, (f_best - mu) is negative. Phi((f_best - mu)/sigma) is small.
       The term (f_best - mu) * Phi(...) will be negative.
       We must clamp EI to be >= 0.
       
    3. Divide by cost to get cEI.
    4. Return cEI.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative. Numerical issues might cause tiny negative values.
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
