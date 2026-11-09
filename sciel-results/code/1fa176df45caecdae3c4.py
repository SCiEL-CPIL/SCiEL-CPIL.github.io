import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on powell 4d.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Predicted mean of the GP at the 2000 candidate points
    sigma : array of shape (2000,)
        Predicted standard deviation of the GP at the 2000 candidate points
    f_best : float
        Best (lowest) observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled)
    
    Returns:
    --------
    score : array of shape (2000,)
        Acquisition scores (higher = prefer to evaluate next)
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Standard EI formula
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic exploration weight based on remaining budget
    # The total budget is 30, starting with 5 initial points (cost ~5-10)
    # We estimate remaining budget from the current cost values
    # cost is the cost of evaluating each candidate, but we need to infer how much budget remains
    
    # Heuristic: estimate remaining budget from the spread of costs
    # If costs are low, we've likely spent less; if high, more
    # Actually, we don't have direct access to total spent, so we use a proxy:
    # The mean cost gives us a sense of where we are in the search
    
    # Better approach: use the ratio of f_best to a reference to estimate progress
    # Or simply use a schedule based on the number of evaluations implied by cost
    
    # Simple dynamic schedule: 
    # - Early: high exploration (beta ~ 2.0)
    # - Late: low exploration (beta ~ 0.3)
    # We estimate progress from the minimum possible cost vs current typical cost
    
    # Since cost = 1 + mean(x_scaled), and x_scaled is in [0,1],
    # cost is in [1, 2]. The initial 5 points have random costs.
    # Total budget is 30. If we've spent C_total, remaining is 30 - C_total.
    # We don't know C_total directly, but we can estimate from f_best improvement rate
    # or use a simpler heuristic based on the typical cost level.
    
    # Let's use the fact that if we're still finding improvements, we have budget left.
    # A simpler approach: assume we're roughly in the middle of the budget and adapt
    # based on the relative uncertainty.
    
    # Actually, let's use a time-based proxy since we can track elapsed time
    # But the function is called each iteration, so we can use a static schedule
    # based on the assumption that we have ~25 evaluations left on average.
    
    # Better: use the sigma distribution to estimate progress
    # High overall sigma = early, low = late
    mean_sigma = np.mean(sigma_safe)
    max_sigma = np.max(sigma_safe)
    
    if max_sigma > 1e-10:
        sigma_ratio = mean_sigma / max_sigma
        # sigma_ratio near 1 = uniform uncertainty (early)
        # sigma_ratio near 0 = concentrated uncertainty (late)
        # Map to beta: early=2.0, late=0.3
        beta = 0.3 + 1.7 * sigma_ratio
    else:
        beta = 0.3
    
    # Cost-aware scoring
    # cost ranges from 1 to 2
    # We want to prefer lower cost evaluations
    
    # Primary: EI normalized by cost
    base_score = ei / cost
    
    # Exploration component: high sigma regions are worth exploring
    # But penalize high cost
    exploration_score = beta * sigma_safe / cost
    
    # Combine
    score = base_score + exploration_score
    
    # Additional bias: prefer regions closer to the expected minimizer
    # The minimizer is in [0.1, 0.9]^4 in scaled coordinates
    # Expected cost at minimizer: 1 + 0.5 = 1.5
    # Add a Gaussian weight centered at cost=1.5
    cost_center = 1.5
    cost_width = 0.25
    cost_weight = np.exp(-0.5 * ((cost - cost_center) / cost_width) ** 2)
    
    # Multiply by a factor that boosts the center region
    multiplier = 0.7 + 0.3 * cost_weight
    score = score * multiplier
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time budget - safety margin before 300s limit
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Return a simpler score to avoid timeout
        simple_score = ei / cost
        simple_score = np.nan_to_num(simple_score, nan=0.0, posinf=0.0, neginf=0.0)
        return simple_score
    
    return score
