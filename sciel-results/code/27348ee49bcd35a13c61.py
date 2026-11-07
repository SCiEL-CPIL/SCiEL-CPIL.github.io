"""Rewritten program using a blend of Expected Improvement and Upper Confidence Bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for Bayesian Optimization.
    Uses a weighted combination of Normalized Expected Improvement (EI) 
    and Normalized Upper Confidence Bound (UCB) to balance exploitation 
    and exploration.
    
    Args:
        mu: Array of mean predictions, shape (2000,)
        sigma: Array of standard deviation predictions, shape (2000,)
        f_best: Current best observed function value (minimum)
        
    Returns:
        Array of scores, shape (2000,). Higher scores are preferred.
    """
    # Ensure sigma is non-negative and not zero to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy.stats.norm for CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score, so we want low mu and high sigma.
    # UCB_score = -mu + beta * sigma
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # 3. Normalize scores to [0, 1] to make them comparable
    def normalize_minmax(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.full_like(s, 0.5) # Return neutral score if no variance
        return (s - s_min) / (s_max - s_min)
    
    ei_norm = normalize_minmax(ei)
    ucb_norm = normalize_minmax(ucb)
    
    # 4. Combine scores
    # Weighting: 60% Exploitation (EI), 40% Exploration (UCB)
    # This helps escape local optima in multimodal functions like Styblinski-Tang
    alpha = 0.6
    scores = alpha * ei_norm + (1.0 - alpha) * ucb_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
