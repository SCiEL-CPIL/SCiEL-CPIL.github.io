"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement with cost discounting.
    
    Uses the Expected Improvement (EI) criterion, which is known to be
    more robust than UCB for multi-modal functions. The EI value is
    discounted by the evaluation cost to favor cheaper evaluations.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t0 = time.time()
    
    # Safety check for empty or invalid inputs
    if len(mu) == 0:
        return np.array([])
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute the improvement: how much better than f_best could we get?
    # For minimization, improvement = f_best - mu
    improvement = f_best - mu
    
    # Standardized improvement
    z = improvement / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    # But this formula is for maximization. For minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Actually, let's use the standard EI formula for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    #       = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute EI
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware discounting: divide by cost to favor cheaper evaluations
    # Using a moderate cost sensitivity
    cost_safe = np.maximum(cost, 1.0)
    
    # Use a power law for cost discounting: 1 / cost^gamma
    # gamma=1.0 is linear, gamma=2.0 is quadratic
    # From observations, gamma=2.0 showed improvement
    gamma = 2.0
    cost_discount = cost_safe ** gamma
    
    # Final score: EI discounted by cost
    score = ei / cost_discount
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 250:
        pass
    
    return score
    # EVOLVE-BLOCK-END
