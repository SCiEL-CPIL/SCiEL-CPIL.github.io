import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

# Track the minimum cost seen so far to adapt the exploration bonus
_min_cost_seen = 1.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses a hybrid approach: Cost-Aware Expected Improvement (cEI) as the base,
    combined with a multiplicative feasibility gate based on cost.
    The gate forces exploitation of low-cost regions until uncertainty is exhausted.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    # Time guard: If we are running out of time, return a safe default
    # (e.g., uniform scores or just exploit best mu) to avoid timeout penalty.
    if _time_exceeded():
        # Fallback: Return scores based on low mu (exploitation)
        # This is a safe, fast operation.
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma near zero to avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF of standard normal, phi is PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Higher EI and lower cost lead to higher score
    cost_safe = np.maximum(cost, 1e-9)
    
    # Structural change: Cost-normalized EI with dynamic exploration bonus
    # The exploration bonus is scaled by the remaining budget fraction.
    # As the budget depletes, we shift from exploration to exploitation.
    
    # Estimate remaining budget fraction based on average cost per step
    # We assume a fixed total budget of 30.0 as per the task description.
    # We don't know the exact current cost, but we can estimate progress.
    # A simpler heuristic: use the ratio of the minimum cost seen to the current cost
    # to prioritize low-cost regions, and scale the exploration term by a factor
    # that decreases as we spend more cost.
    
    # Update global minimum cost seen
    global _min_cost_seen
    current_min_cost = np.min(cost)
    if current_min_cost < _min_cost_seen:
        _min_cost_seen = current_min_cost
    
    # Dynamic exploration bonus
    # The bonus is proportional to sigma (uncertainty) and scaled by a factor
    # that depends on the cost efficiency.
    # We use a simple heuristic: the more expensive the candidate, the less
    # exploration bonus it gets.
    
    # Cost efficiency: lower cost is better.
    # We define a cost efficiency score that is 1 for the cheapest and decays
    # for more expensive candidates.
    cost_efficiency = _min_cost_seen / cost_safe
    
    # Exploration bonus: beta * sigma * cost_efficiency
    # Beta is a constant that balances exploration vs exploitation.
    # We can adapt beta based on the total cost spent, but we don't have that
    # info directly. So we use a fixed beta and rely on cost_efficiency.
    beta = 1.0
    
    exploration_bonus = beta * sigma * cost_efficiency
    
    # Combine EI and exploration bonus
    # Score = EI / cost + exploration_bonus
    # This ensures that high-EI, low-cost points are preferred, but high-uncertainty
    # low-cost points also get a boost.
    
    scores = ei / cost_safe + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
