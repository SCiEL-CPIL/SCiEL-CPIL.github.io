# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses a hybrid of Probability of Improvement (PI) and Expected Improvement (EI)
    to balance exploration and exploitation, normalized by cost.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-9)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability in CDF/PDF
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z_clipped)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # For minimization, EI is positive when mu < f_best
    # We want to maximize the acquisition score (higher = evaluate next)
    # Combine PI and EI: use a weighted sum or product
    # In multimodal Rastrigin, PI is more robust to high uncertainty regions
    # Use a blend: weight PI more when uncertainty is high
    
    # Normalize EI to be comparable to PI
    # EI can be large when f_best - mu is large, so scale it
    # Use a soft combination: score = pi * (1 + alpha * ei_scaled)
    # where ei_scaled is bounded
    
    # Alternative: use qEI (quantile-based) or just PI with cost normalization
    # Given the finding that PI improved score from 1.392 to 3.173, emphasize PI
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    # cost = 1 + mean(x_scaled), so cost in [1, 2]
    
    # Primary: PI / cost
    # Secondary: add a small EI component for exploitation
    # Use a geometric mean or weighted sum
    
    # Let's use: score = (pi^0.7 * (1 + 0.1 * ei)^0.3) / cost
    # This emphasizes PI but retains some EI signal
    
    # Simpler: score = (pi + 0.1 * ei / (1 + abs(ei))) / cost
    # The term 0.1 * ei / (1 + abs(ei)) bounds the EI contribution
    
    ei_bounded = 0.1 * ei / (1.0 + np.abs(ei))
    
    score = (pi + ei_bounded) / np.maximum(cost, 1e-9)
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
