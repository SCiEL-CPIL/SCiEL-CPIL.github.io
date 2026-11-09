"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using an adaptive ensemble of EI and UCB strategies.
    Uses a soft-combination to balance exploration and exploitation based on
    the uncertainty (sigma) of the GP posterior.
    """
    # Handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    # Improvement is defined as f_best - f(x) if f(x) < f_best, else 0.
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to minimize f. The optimistic bound is mu - beta * sigma.
    # We want to maximize the score, so score = -(mu - beta * sigma) = beta * sigma - mu.
    # We use a slightly higher beta than the baseline to encourage exploration
    # of the multimodal Hartmann function, but not so high that we ignore exploitation.
    beta = 2.5
    ucb_scores = beta * sigma_safe - mu
    
    # Normalize scores to be comparable (Z-score normalization)
    def normalize(scores):
        m = np.mean(scores)
        s = np.std(scores)
        if s < 1e-9:
            return scores - m
        return (scores - m) / s
        
    z_ei = normalize(ei_scores)
    z_ucb = normalize(ucb_scores)
    
    # Adaptive Weighting:
    # When uncertainty (sigma) is high, we trust UCB more (explore).
    # When uncertainty is low, we trust EI more (exploit).
    # avg_sigma indicates the current level of uncertainty in the sampled candidates.
    avg_sigma = np.mean(sigma_safe)
    
    # Scale factor to control the transition between exploration and exploitation.
    # If avg_sigma is small, w_exploit should be close to 1.
    # If avg_sigma is large, w_exploit should be close to 0.
    # We use a sigmoid-like function.
    # Let's define a threshold. In [0,1]^3 Hartmann, sigma is usually < 0.5.
    # Let's map avg_sigma to a weight.
    # w_exploit = 1 / (1 + exp(k * (avg_sigma - threshold)))
    # Simpler: linear mapping with clipping.
    # If avg_sigma < 0.01, w_exploit = 0.9
    # If avg_sigma > 0.2, w_exploit = 0.2
    
    # Heuristic: 
    # Low sigma -> Exploit (EI). High sigma -> Explore (UCB).
    # Let's use a simple inverse relationship.
    # w_exploit = 1.0 / (1.0 + 10.0 * avg_sigma)
    # If avg_sigma = 0.01, w = 1/1.1 = 0.9
    # If avg_sigma = 0.1, w = 1/2.0 = 0.5
    # If avg_sigma = 0.5, w = 1/6.0 = 0.16
    
    w_exploit = 1.0 / (1.0 + 10.0 * avg_sigma)
    w_explore = 1.0 - w_exploit
    
    # Combine scores
    final_scores = w_exploit * z_ei + w_explore * z_ucb
    
    return final_scores
# EVOLVE-BLOCK-END
