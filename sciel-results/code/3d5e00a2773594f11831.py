"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid Cost-Aware Acquisition Function: 50% Expected Improvement + 50% UCB.
    
    This structural change moves from the binary-like Probability of Improvement (PI)
    to a continuous Expected Improvement (EI) blended with an Upper Confidence Bound (UCB)
    term. This balances exploitation (EI) and exploration (UCB) more effectively for
    multimodal functions like Griewank, while maintaining cost-awareness via division
    by the evaluation cost.
    
    Args:
        mu: Array of shape (N,) with predicted means.
        sigma: Array of shape (N,) with predicted standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    cost_safe = np.maximum(cost, epsilon)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using norm.cdf and norm.pdf
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # 2. Compute UCB for minimization
    # We want to minimize mu - beta * sigma.
    # To maximize the score, we use: beta * sigma - mu
    beta = 2.5  # Exploration parameter
    ucb = beta * sigma_safe - mu
    
    # 3. Blend EI and UCB
    # 50/50 blend as suggested by successful inspirations
    w_ei = 0.5
    w_ucb = 0.5
    combined_score = w_ei * ei + w_ucb * ucb
    
    # 4. Cost-awareness: Divide by cost
    # This penalizes expensive evaluations
    scores = combined_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
