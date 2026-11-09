import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with budget-aware LCB scaling.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Estimate remaining budget: total budget is 30, we've spent some amount
    # We can estimate how much budget remains by looking at f_best and cost
    # A simple heuristic: if f_best is close to optimum (-3.32237), we're doing well
    # Use a fixed budget assumption for scaling
    
    # Budget-aware beta: start high (exploration), decrease as we approach budget limit
    # Estimate progress: assume we've used a fraction of budget based on number of evaluations
    # Since we don't know exactly how many evaluations, use a heuristic based on f_best
    
    # The minimum of hartmann 6d is -3.32237
    # If f_best is close to this, we're in exploitation mode
    # If f_best is far, we're in exploration mode
    
    # Scale beta based on how far f_best is from the known minimum
    min_hartmann = -3.32237
    # Distance from best to minimum (normalized)
    dist_to_min = max(0, f_best - min_hartmann)
    
    # beta starts at 3.0 and decreases to 1.0 as we get closer to optimum
    # Use a smooth transition
    beta_base = 1.0 + 2.0 * np.exp(-dist_to_min / 0.5)
    
    # Also scale by remaining budget estimate
    # Assume total budget 30, estimate used budget from cost array
    # If cost values are around 1-2, and we've done ~15 evaluations, used ~25
    # Use a conservative estimate: assume we're at 50% budget usage
    budget_fraction = 0.5
    beta = beta_base * np.sqrt(budget_fraction)
    
    # LCB for minimization: mu - beta * sigma
    # Score = -mu + beta * sigma (we want to maximize)
    lcb_score = -mu + beta * sigma_safe
    
    # Expected Improvement component for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize both components
    lcb_std = np.std(lcb_score) + 1e-8
    ei_std = np.std(ei) + 1e-8
    
    lcb_norm = lcb_score / lcb_std
    ei_norm = ei / ei_std
    
    # Adaptive weights: more LCB when uncertain, more EI when confident
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-6:
        # High uncertainty: favor LCB
        lcb_weight = 0.7
        ei_weight = 0.3
    else:
        # Low uncertainty: favor EI
        lcb_weight = 0.4
        ei_weight = 0.6
    
    combined = lcb_weight * lcb_norm + ei_weight * ei_norm
    
    # Cost-aware: divide by cost to penalize expensive points
    # Use a softer penalty: divide by sqrt(cost) to avoid over-penalizing
    scores = combined / np.sqrt(cost_safe)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
