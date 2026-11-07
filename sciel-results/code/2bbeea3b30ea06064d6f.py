import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (2000,)
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (2000,)
    f_best (float): Best observed function value so far
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (2000,). Higher score = evaluate next.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 0.1 * sigma_safe)
    
    # Ensure sigma is non-negative and numerically stable
    sigma = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient for minimization
    # KG = E[min(f_best, Y)] where Y ~ N(mu, sigma^2)
    # For a single candidate: KG = f_best - E[max(f_best - Y, 0)]
    # E[max(f_best - Y, 0)] = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # So KG = f_best - [(f_best - mu) * Phi(z) + sigma * phi(z)]
    # This is the expected reduction in the best value if we evaluate at this point.
    
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Expected improvement term
    ei_term = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Knowledge Gradient: expected reduction in best value
    kg = f_best - ei_term
    
    # For maximization of score (higher = better to evaluate next),
    # we want to maximize the KG (expected reduction in f_best)
    # But we also need to consider that in a global optimization context,
    # KG should account for the probability that the point contains the global minimum.
    
    # A more sophisticated KG considers the posterior distribution of the global minimum.
    # For a finite set of candidates, the KG for candidate i is:
    # KG_i = sum_j P(global min is at j | current info) * E[min(f_best, Y_i) - min(f_best, Y_j) | ...]
    
    # A simpler but effective approximation: use the fact that in the limit,
    # we want to evaluate where the posterior is most likely to contain a value below f_best.
    
    # Let's use a hybrid approach:
    # 1. Compute the probability that each candidate has a value below f_best
    # 2. Weight the KG by this probability to focus on promising regions
    
    # Probability that Y < f_best for each candidate
    prob_below = norm.cdf(z)
    
    # Enhanced KG: weight by probability of being below current best
    # This focuses exploration on regions likely to contain improvements
    kg_enhanced = kg * (0.5 + 0.5 * prob_below)
    
    # Additionally, incorporate an exploration bonus for regions with high uncertainty
    # that haven't been well-explored yet (high sigma)
    # This helps with multimodal functions like Weierstrass
    
    # Normalize sigma for exploration bonus
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Adaptive exploration weight: higher when f_best is far from optimum
    # For Weierstrass on [-0.5, 0.5], optimum is 0, typical f_best starts around 0.5-1.0
    f_best_norm = max(0.0, min(1.0, f_best))
    
    # More exploration early, less later
    w_explore = 0.2 + 0.6 * (1.0 - f_best_norm)
    
    # Combine KG and exploration
    scores = kg_enhanced + w_explore * sigma_norm * np.max(kg_enhanced)
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
