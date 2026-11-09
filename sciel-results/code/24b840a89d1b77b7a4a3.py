import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining PI/cost ranking with adaptive
    cost exponent and mild EI exploitation.
    
    Primary signal: rank-normalized Probability of Improvement per unit cost
    with adaptive cost exponent that sharpens penalty near convergence.
    Refinement: small EI component to exploit regions where mean < f_best.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    n = len(mu)
    
    # Compute Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Handle sigma near zero
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * pi + sigma_safe * phi_z
    
    # Rank-based normalization for robustness
    # PI rank: higher PI is better
    rank_pi = np.argsort(np.argsort(-pi))
    
    # EI rank: higher EI is better  
    rank_ei = np.argsort(np.argsort(-ei))
    
    # Cost: lower cost is better
    rank_cost = np.argsort(np.argsort(cost))
    
    # Adaptive cost exponent: sharpen cost penalty as we converge
    # When f_best is small (close to optimum), penalize high cost more
    # f_best for rosenbrock starts around 1000+ and goes to 0
    # Use a smooth transition
    f_best_scaled = max(f_best, 1e-6)
    # As f_best decreases, increase cost sensitivity
    # cost_exp ranges from 0.5 (early) to 2.0 (late)
    cost_exp = 0.5 + 1.5 * (1.0 / (1.0 + f_best_scaled))
    
    # Cost-adjusted score using ranks
    # Combine PI and EI with cost adjustment
    # Use rank-based approach for stability
    w_pi = 0.6
    w_ei = 0.4
    
    # Base score from PI and EI
    base_score = w_pi * (n - 1 - rank_pi) + w_ei * (n - 1 - rank_ei)
    
    # Apply cost penalty using the adaptive exponent
    # Normalize cost to [0, 1] for consistent scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost penalty: higher cost gets lower score
    # Use the adaptive exponent to sharpen the penalty
    cost_penalty = cost_norm ** cost_exp
    
    # Final score: base_score minus cost penalty
    scores = base_score - cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
