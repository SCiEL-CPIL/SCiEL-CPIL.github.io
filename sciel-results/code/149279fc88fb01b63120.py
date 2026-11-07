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

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute z-scores: how many standard deviations below f_best the mean is
    z = (f_best_val - mu) / sigma_safe
    # Clip z for numerical stability in CDF/PDF evaluation
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best_val - mu) * Phi_z + sigma_safe * phi_z
    # EI should be non-negative; clip tiny negative values from numerical noise
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PI):
    # PI = Phi(z) = P(f(x) < f_best)
    pi = Phi_z
    
    # Adaptive mixing based on proximity to optimum:
    # Griewank 2D values are in [0, 1]. 
    # When f_best is close to 0, exploit heavily (weight PI and EI).
    # When f_best is large, explore more (weight sigma-based exploration).
    f_best_clipped = np.clip(f_best_val, 0.0, 1.0)
    
    # Weights: as f_best → 0, exploit_weight → 1, explore_weight → 0
    exploit_weight = 1.0 - f_best_clipped
    explore_weight = f_best_clipped
    
    # Cost-adjusted component scores
    ei_score = ei / cost_safe
    pi_score = pi / cost_safe
    explore_score = sigma_safe / cost_safe
    
    # Adaptive combination weights:
    # w_ei: primary exploitation via EI, boosted when close to optimum
    # w_pi: secondary pure-confidence exploitation
    # w_explore: tertiary exploration via uncertainty, active when far from optimum
    w_ei = 0.6 + 0.4 * exploit_weight
    w_pi = 0.2 * exploit_weight
    w_explore = 0.2 * explore_weight
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_pi + w_explore
    w_ei /= w_sum
    w_pi /= w_sum
    w_explore /= w_sum
    
    scores = w_ei * ei_score + w_pi * pi_score + w_explore * explore_score
    
    # Deadline guard: if we're running too long, fall back to simple exploitation
    if time.time() > deadline:
        scores = -mu / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
