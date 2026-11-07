"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement with adaptive exploration
    and quadratic cost penalty.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Normalize
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid overflow in exp
    z_clipped = np.clip(z, -10, 10)
    
    # Compute EI
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Handle case where mu > f_best (unlikely to improve)
    ei = np.maximum(ei, 0.0)
    
    # Add small exploration bonus based on sigma to avoid pure exploitation
    # This helps in early iterations when f_best is not yet well-estimated
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration
    score = ei + exploration_bonus
    
    # Cost penalty: divide by cost^2 to strongly prefer cheap evaluations
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = cost_safe ** 2
    
    # Final score: (EI + exploration) / cost^2
    scores = score / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
