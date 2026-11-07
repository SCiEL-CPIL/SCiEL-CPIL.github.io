"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with cost normalization and adaptive exploration.
    
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

    # Ensure sigma is positive and cost is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute z-scores: how many standard deviations below f_best the mean is
    z = (f_best_val - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best_val - mu) * Phi_z + sigma_safe * phi_z
    
    # Probability of Improvement (PI):
    pi = Phi_z
    
    # Lower Confidence Bound (LCB) for minimization:
    # LCB = mu - beta * sigma
    # Use a dynamic beta that decreases as f_best approaches 0
    f_best_clipped = np.clip(f_best_val, 0.0, 1.0)
    beta = 0.5 + 1.5 * f_best_clipped  # Higher exploration when f_best is large
    lcb = mu - beta * sigma_safe
    
    # Cost-aware scores
    ei_score = ei / cost_safe
    pi_score = pi / cost_safe
    lcb_score = -lcb / cost_safe  # Negate because we want to minimize LCB
    explore_score = sigma_safe / cost_safe
    
    # Adaptive weights based on f_best
    # When f_best is close to 0, exploit more (EI, PI)
    # When f_best is far from 0, explore more (LCB, sigma)
    exploit_weight = 1.0 - f_best_clipped
    explore_weight = f_best_clipped
    
    # Combine scores with adaptive weights
    w_ei = 0.5 + 0.3 * exploit_weight
    w_pi = 0.2 * exploit_weight
    w_lcb = 0.2 + 0.3 * explore_weight
    w_explore = 0.1 * explore_weight
    
    # Normalize weights
    w_sum = w_ei + w_pi + w_lcb + w_explore
    w_ei /= w_sum
    w_pi /= w_sum
    w_lcb /= w_sum
    w_explore /= w_sum
    
    scores = w_ei * ei_score + w_pi * pi_score + w_lcb * lcb_score + w_explore * explore_score
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple cost-aware EI
        scores = ei_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
