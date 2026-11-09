"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for each candidate.
    
    Uses a variant of Expected Improvement (EI) adjusted for cost.
    The key insight from the knowledge is that PI (Probability of Improvement)
    with cost^0.5 normalization can be smoother and better for exploration.
    
    We'll use a hybrid approach:
    - Compute EI as the baseline
    - Blend with PI for smoother gradients in multimodal landscapes
    - Use cost^0.5 normalization for gentler cost penalty
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Blend EI and PI for smoother gradients
    # Use a weighted combination where PI helps in multimodal regions
    # The blend factor can be adaptive based on uncertainty
    # Higher uncertainty -> more weight to PI for exploration
    blend_weight = 0.3  # Weight for PI component
    
    # Cost-aware normalization using cost^0.5 for gentler penalty
    # This allows better exploration of low-cost regions early
    cost_norm = np.sqrt(cost)
    
    # Combine EI and PI with cost normalization
    # Scale EI by a factor to balance with PI
    ei_scaled = ei / (np.max(ei) + 1e-9)
    
    # Final score: blend of normalized EI and PI, divided by cost^0.5
    scores = (ei_scaled + blend_weight * pi) / cost_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
