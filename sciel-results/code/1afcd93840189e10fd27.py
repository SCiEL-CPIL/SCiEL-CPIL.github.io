"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: Expected Improvement per unit cost,
    with a budget-aware penalty that increases late in the search.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Track elapsed time for safety margin
    t_start = time.time()
    deadline = 280.0  # seconds, leaving 50s margin before 330s limit
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Budget-aware penalty: as total cost approaches 30, penalize expensive points more
    # Estimate remaining budget from mean cost and number of evaluations
    # We don't know exact remaining budget, but we can use cost as a proxy
    # Late in search (high mean cost or low EI), we should prefer cheaper points
    
    # A smooth transition: early on, pure EI/cost; late on, add uncertainty reduction reward
    # Use the median cost as a reference for "how far along" we are
    cost_median = np.median(cost)
    
    # The penalty term: rewards uncertainty (high sigma) when EI is low
    # This helps exploration early and exploitation late
    # Weight the uncertainty term based on how "expensive" the candidates are
    # relative to the cheapest ones
    
    # Soft-min style: combine EI/cost with a KG-like term
    # KG = sigma * (z * cdf(z) + pdf(z)) / (f_best - mu + eps)
    # But simpler: just add a scaled sigma term that fades as EI grows
    
    # Scale factor: 1 when EI is small (explore), 0 when EI is large (exploit)
    ei_scale = np.maximum(ei, 1e-9)
    explore_weight = 1.0 / (1.0 + ei_scale * 10.0)  # sigmoid-like decay
    
    # Uncertainty bonus: reward high sigma points, weighted by how cheap they are
    sigma_bonus = sigma_safe / cost_safe
    bonus_weight = 0.1 * explore_weight  # small weight for exploration
    
    # Combined score
    scores = ei_per_cost + bonus_weight * sigma_bonus
    
    # Check time
    if time.time() - t_start > deadline:
        # Return best result found so far (already computed)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
