import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a dynamic Lower Confidence Bound (LCB).
    
    Args:
        mu: Mean of the GP posterior (shape: (2000,))
        sigma: Standard deviation of the GP posterior (shape: (2000,))
        f_best: Best observed value so far (scalar)
        cost: Cost of evaluating each candidate (shape: (2000,))
        
    Returns:
        scores: Acquisition scores (shape: (2000,)), higher is better
    """
    # Ensure inputs are safe
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Handle NaNs and infinities
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=-np.inf)
    sigma_safe = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Ensure sigma is positive to avoid division by zero in z-scores if we used them,
    # though LCB doesn't strictly require division by sigma, it's good practice.
    sigma_safe = np.maximum(sigma_safe, 1e-8)
    cost_safe = np.maximum(cost_safe, 1e-8)
    
    # Dynamic exploration coefficient beta
    # We estimate the number of evaluations based on the total cost budget.
    # The task states the loop runs while total cost < 30.
    # Initial cost is ~2.5 (5 points * avg cost 1.5).
    # Each subsequent eval costs ~1.5 on average.
    # Total evals N ≈ (30 - 2.5) / 1.5 ≈ 18 + 5 = 23.
    # We can approximate the current evaluation index 't' by assuming a linear progression.
    # However, we don't have 't' directly. We can infer it from f_best or just use a fixed schedule.
    # A common heuristic is beta = sqrt(2 * log(t * pi^2 / (6 * alpha)))
    # Let's assume a total of ~25 evaluations.
    # To make it adaptive without state, we can use the magnitude of f_best relative to the domain?
    # No, that's unreliable. 
    # Alternative: Use a fixed but slightly aggressive beta that works well for small N.
    # For N=25, beta ≈ sqrt(2 * log(25 * 1.645)) ≈ sqrt(2 * 3.5) ≈ 2.6.
    # Let's use a beta that starts high and decays? We can't decay without state.
    # Let's use a constant beta that is tuned for small sample sizes, e.g., beta = 2.0.
    # Or better, use a beta that depends on the uncertainty itself? No, LCB is linear in sigma.
    
    # Let's stick to a standard LCB with a fixed beta that is effective for small N.
    # beta = 2.0 is a reasonable default for N < 50.
    beta = 2.0
    
    # LCB for minimization: minimize mu - beta * sigma
    # We want to maximize the score, so score = -(mu - beta * sigma) = -mu + beta * sigma
    # This encourages low mu (exploitation) and high sigma (exploration).
    
    lcb_score = -mu_safe + beta * sigma_safe
    
    # Cost adjustment: Divide by cost to prioritize cheaper evaluations.
    # This is the "cost-aware" part.
    scores = lcb_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Shift scores to be positive for stability, though not strictly necessary
    # since we only care about argmax.
    
    return scores
