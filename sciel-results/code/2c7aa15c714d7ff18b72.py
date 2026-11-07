"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid Expected Improvement (EI)
    and Probability of Improvement (PI) strategy with adaptive exploration.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with scores. Higher score is better.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    f_best_val = float(f_best)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute z-scores: how many standard deviations below f_best the mean is
    # z = (f_best - mu) / sigma
    z = (f_best_val - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best_val - mu) * Phi_z + sigma_safe * phi_z
    
    # Probability of Improvement (PI):
    # PI = Phi(z) = P(f(x) < f_best)
    pi = Phi_z
    
    # Adaptive mixing:
    # When f_best is close to 0 (good), exploit more (weight EI/PI heavily)
    # When f_best is far from 0 (bad), explore more (use sigma-based exploration)
    # For Griewank, values are in [0, 1], typically starting near 0.5-1.0
    f_best_clipped = np.clip(f_best_val, 0.0, 1.0)
    
    # Blend between pure exploitation (PI) and exploration (sigma)
    # As f_best approaches 0, we want to exploit: weight toward PI/EI
    # As f_best is large, we want to explore: weight toward sigma
    exploit_weight = 1.0 - f_best_clipped  # High when f_best is small
    explore_weight = f_best_clipped        # High when f_best is large
    
    # Combine EI and PI with exploration
    # EI already includes both exploitation and exploration components
    # PI is pure exploitation
    # sigma is pure exploration
    
    # Score = (exploit_weight * EI + explore_weight * sigma * 0.5) / cost
    # But EI can be negative if mu > f_best, which is fine (we won't pick those)
    # Let's use a more robust combination:
    
    # Primary: EI / cost (standard cost-aware EI)
    ei_score = ei / cost_safe
    
    # Secondary: PI / cost (probability of improvement per unit cost)
    pi_score = pi / cost_safe
    
    # Tertiary: sigma / cost (pure exploration per unit cost)
    explore_score = sigma_safe / cost_safe
    
    # Adaptive combination:
    # weight_ei = 0.5 + 0.5 * exploit_weight  (0.5 to 1.0)
    # weight_pi = 0.3 * exploit_weight        (0 to 0.3)
    # weight_explore = 0.2 * explore_weight   (0 to 0.2)
    
    w_ei = 0.6 + 0.4 * exploit_weight
    w_pi = 0.2 * exploit_weight
    w_explore = 0.2 * explore_weight
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_pi + w_explore
    w_ei /= w_sum
    w_pi /= w_sum
    w_explore /= w_sum
    
    scores = w_ei * ei_score + w_pi * pi_score + w_explore * explore_score
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple cost-aware exploitation
        scores = -mu / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
