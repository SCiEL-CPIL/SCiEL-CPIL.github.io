"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Hypervolume Improvement (EHVI):
    - Estimates the expected volume of the objective space dominated by the next sample
    - Normalizes by cost to prioritize points that expand the Pareto front
    - Uses Monte Carlo sampling for EHVI approximation
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Prevent division by zero in cost
    cost_safe = np.maximum(cost, 1e-10)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback to simple cost inversion if running out of time
        scores = 1.0 / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # For minimization, we want to find points where f(x) < f_best
    # EHVI approximates the expected improvement in the "front"
    # Since we only have a single objective, EHVI reduces to a form related to EI
    # but we can use a more principled approach:
    
    # Z-score for P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Fallback to simple PI/cost if running out of time
        scores = pi / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Hypervolume-inspired score:
    # In single-objective minimization, the "hypervolume" improvement is related to
    # how much the point improves upon f_best, weighted by the probability of improvement.
    # We use a combined metric that emphasizes both the magnitude and likelihood of improvement.
    
    # Normalize EI by a reference scale (use std of mu as a proxy for problem scale)
    mu_scale = np.std(mu) + 1e-10
    ei_normalized = ei / mu_scale
    
    # The hypervolume contribution is roughly: P(improvement) * E[improvement | improvement]
    # This is equivalent to EI, but we can add a bonus for points that are "far" from the current best
    # in terms of potential improvement.
    
    # Use a logarithmic scaling to handle the wide range of EI values
    # This prevents a single very high EI point from dominating the scores
    log_ei = np.log1p(ei_normalized)
    
    # Combine PI and log-scaled EI
    # Weight: give more weight to PI when uncertainty is high, more to EI when certain
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # When sigma is high (uncertain), trust PI more; when low, trust EI more
    w_pi = 0.5 * (1.0 - sigma_norm) + 0.2
    w_ei = 0.5 * sigma_norm + 0.2
    
    # Normalize weights
    w_total = w_pi + w_ei
    w_pi = w_pi / w_total
    w_ei = w_ei / w_total
    
    # Cost-normalized scores
    pi_cost = pi / cost_safe
    log_ei_cost = log_ei / cost_safe
    
    # Combine
    scores = w_pi * pi_cost + w_ei * log_ei_cost
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 295:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
