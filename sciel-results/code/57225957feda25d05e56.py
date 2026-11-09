"""Pareto-based multi-objective acquisition for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Dynamic weighted sum of normalized Probability of Improvement (PI) and 
    normalized predictive uncertainty.
    
    PI is more robust to scale variations than EI. The weight on exploitation (PI)
    increases as the maximum available EI decreases, indicating convergence.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std dev in objective units
        f_best: float - best observed value so far (lowest)
    
    Returns:
        scores: array of shape (2000,) - higher scores mean evaluate next
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute Probability of Improvement (PI) and Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # PI is the probability that f(x) < f_best
    pi = norm.cdf(z)
    
    # EI is used to determine the dynamic weight (how much improvement is possible)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Predictive Uncertainty
    uncertainty = sigma
    
    # 3. Determine Dynamic Weight for Exploitation
    # If the best possible EI is small, we are close to the optimum -> Exploit
    # If the best possible EI is large, we are far from the optimum -> Explore
    max_ei = np.max(ei)
    
    # Scale parameter C controls the transition speed.
    # Increased C to 0.5 to allow for a smoother transition and more exploration
    # in the mid-game.
    C = 0.5
    
    # Weight on exploitation (PI)
    # w_exp is high when max_ei is low.
    w_exploit = 1.0 / (1.0 + (max_ei / C) ** 2)
    
    # Weight on exploration (Uncertainty)
    w_explore = 1.0 - w_exploit
    
    # 4. Normalize Objectives
    # PI is already in [0, 1].
    # Uncertainty is normalized to [0, 1] by its max.
    
    max_unc_val = np.max(uncertainty)
    
    if max_unc_val < 1e-9:
        unc_norm = np.zeros_like(uncertainty)
    else:
        unc_norm = uncertainty / max_unc_val
        
    # 5. Combine Objectives
    # PI is already normalized to [0, 1]
    score = w_exploit * pi + w_explore * unc_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
