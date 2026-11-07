import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses Expected Improvement (EI) with a multiplicative cost gate.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative (EI should always be >= 0 theoretically, but numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware gate: only consider candidates with cost below a threshold
    # The threshold is set to the mean cost of the top 10% EI candidates
    # This creates a hard budget-aware gate
    
    # Sort by EI to find the threshold
    ei_sorted = np.sort(ei)[::-1]
    top_10_percent_idx = max(1, int(0.1 * len(ei)))
    top_ei_threshold = ei_sorted[top_10_percent_idx - 1]
    
    # Get costs of top EI candidates
    top_candidates_mask = ei >= top_ei_threshold
    if np.any(top_candidates_mask):
        top_costs = cost[top_candidates_mask]
        cost_threshold = np.mean(top_costs)
    else:
        cost_threshold = np.mean(cost)
    
    # Apply the gate: only score candidates with cost <= cost_threshold
    # For candidates above the threshold, give them a very low score
    gate = (cost <= cost_threshold).astype(float)
    
    # Final score: EI multiplied by the gate, with a small penalty for higher costs within the gate
    # Use a smooth transition instead of hard gate for better optimization
    cost_penalty = 1.0 / (1.0 + 0.5 * np.maximum(cost - cost_threshold, 0.0) / np.maximum(cost_threshold, 1e-9))
    
    scores = ei * gate * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
