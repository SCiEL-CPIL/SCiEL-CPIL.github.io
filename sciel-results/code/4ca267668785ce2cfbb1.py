# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 330.0  # seconds
_START = time.time()


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better).
    """
    # Safety margin: if we're close to deadline, return quick result
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: simple EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB term with adaptive beta
    # Use a beta that balances exploration and exploitation
    # Early on, explore more; later, exploit more
    # We can estimate progress from f_best relative to expected minimum (0 for ackley)
    # But since we don't know the instance, use a heuristic based on cost spent
    
    # Adaptive UCB: beta increases as we spend more budget
    # cost has shape (2000,), but we can use the mean cost as a proxy for current budget usage
    mean_cost = np.mean(cost)
    
    # Estimate how much budget we've used: f_best is the best value, but we don't have total cost directly
    # Instead, use a fixed exploration schedule based on the number of evaluations implied
    # Since cost = 1 + mean(x_scaled), and we start with 5 points, each costing ~1.5 on average
    # Total budget is 30, so we can have about 30/1.5 = 20 evaluations
    
    # Use a simple adaptive beta: start high, decrease over time
    # We can use the fact that mu and sigma give us information about the GP
    # A common heuristic: beta = 2 * sqrt(log(1 + n)) where n is number of evals
    # But we don't have n directly. Use cost as a proxy.
    
    # Alternative: use a fixed beta that works well in practice
    beta = 1.5
    
    ucb = mu + beta * sigma
    
    # Combine EI and UCB: use a weighted sum
    # Normalize both to comparable scales
    ei_max = np.max(ei) + 1e-9
    ucb_max = np.max(np.abs(ucb)) + 1e-9
    
    # Weight: more weight to EI when f_best is close to minimum, more UCB when uncertain
    # Simple approach: geometric mean or weighted harmonic mean
    # Let's use a blend: score = (ei / cost) * (1 + alpha * ucb_normalized)
    # where ucb_normalized is negative for low mu (good)
    
    # Actually, let's think about this differently:
    # We want to maximize the expected improvement per unit cost, but also explore
    # A good approach: score = EI/cost + gamma * UCB/cost
    # where UCB here is the exploration bonus (sigma term)
    
    # Let's use: score = (EI + gamma * sigma) / cost
    # where gamma controls exploration
    gamma = 0.5
    
    # But we need to be careful: EI already includes sigma, so adding more sigma might over-explore
    # Better: use a proper cost-aware UCB
    # score = (f_best - mu + beta * sigma) / cost  (this is like UCB for minimization)
    
    # Let's try a hybrid: 
    # Primary: EI/cost
    # Secondary: exploration bonus for high sigma regions
    ei_cost = ei / (cost + 1e-9)
    
    # Exploration bonus: reward high sigma relative to EI
    # Normalize sigma by the scale of the problem
    sigma_norm = sigma / (np.std(mu) + 1e-9)
    exploration_bonus = gamma * sigma_norm / (cost + 1e-9)
    
    score = ei_cost + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
