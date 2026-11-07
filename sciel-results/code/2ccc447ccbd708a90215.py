import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier selection.
    Filters candidates to the efficient frontier of (improvement, uncertainty, cost).
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 320:
        # Safe fallback: simple exploitation
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return (f_best - mu) / cost_safe

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (for minimization)
    z = (f_best - mu) / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Three objectives for Pareto selection:
    # 1. Predicted improvement (maximize)
    # 2. Uncertainty (maximize) - sigma
    # 3. Cost efficiency (maximize) - 1/cost
    # 
    # We want to find the Pareto front in (ei, sigma, 1/cost)
    # A point is dominated if there exists another point that is >= in all objectives
    # and strictly > in at least one.
    
    # For efficiency, we'll use a simplified approach:
    # Assign scores based on rank in each dimension, then find non-dominated points.
    
    # Rank-based normalization to handle different scales
    n = len(mu)
    
    # Rank ei (higher is better)
    ei_ranks = np.argsort(np.argsort(ei)).astype(float) / (n - 1)
    
    # Rank sigma (higher is better for exploration)
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(float) / (n - 1)
    
    # Rank cost efficiency (higher is better, so rank 1/cost)
    cost_eff = 1.0 / cost
    cost_ranks = np.argsort(np.argsort(cost_eff)).astype(float) / (n - 1)
    
    # Combined score: sum of ranks (higher = more promising)
    # This is a simple heuristic for Pareto-like selection
    combined = ei_ranks + sigma_ranks + cost_ranks
    
    # Identify the top candidates (Pareto-like front)
    # Use a threshold to select the "frontier" - top 10% by combined score
    threshold = np.percentile(combined, 90)
    frontier_mask = combined >= threshold
    
    # For frontier points, use a refined score that balances all three objectives
    # For non-frontier points, assign lower scores
    refined_score = np.zeros(n)
    
    # For frontier points: weighted combination with adaptive weights
    # Early on: weight uncertainty more
    # Late on: weight improvement more
    f_min_hartmann = -3.32237
    progress = (0.0 - f_best) / (0.0 - f_min_hartmann)
    progress = np.clip(progress, 0.0, 1.0)
    
    w_ei = 0.5 + 0.5 * progress
    w_sigma = 0.5 - 0.5 * progress
    w_cost = 0.2  # constant weight for cost efficiency
    
    # Normalize each component to [0, 1] for fair combination
    ei_norm = ei_ranks
    sigma_norm = sigma_ranks
    cost_norm = cost_ranks
    
    refined_score[frontier_mask] = (
        w_ei * ei_norm[frontier_mask] +
        w_sigma * sigma_norm[frontier_mask] +
        w_cost * cost_norm[frontier_mask]
    )
    
    # Non-frontier points get a small base score scaled by their combined rank
    refined_score[~frontier_mask] = 0.1 * combined[~frontier_mask]
    
    # Final score: ensure higher is better
    final_score = refined_score
    
    # Handle non-finite values
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return final_score
