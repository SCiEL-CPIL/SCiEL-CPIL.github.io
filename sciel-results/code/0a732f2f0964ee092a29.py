import numpy as np

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (cosine8 8d).
    
    Uses a cost-normalized Lower Confidence Bound (LCB) with a dynamic exploration
    parameter (kappa) that adapts based on the current best observed value (f_best).
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays for consistent behavior
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-9)
    
    # Sanitize cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic Kappa Calculation
    # We use f_best as a proxy for progress.
    # f_best starts around 0.0 and aims for -0.8.
    f_min_global = -0.8
    f_initial_estimate = 0.0
    
    # Distance from the optimum (larger distance = less progress)
    dist = np.maximum(0.0, f_best - f_min_global)
    # Normalize distance: 0.8 is the max expected initial distance
    max_dist = 0.8
    normalized_dist = np.clip(dist / max_dist, 0.0, 1.0)
    
    # Kappa parameters
    # We reduce the exploration range to avoid over-exploration in 8D.
    # Higher kappa can lead to chasing noise.
    # Linear decay is smoother than quadratic for budget management.
    kappa_low = 0.2
    kappa_high = 1.5
    
    # Linear mapping: 
    # At normalized_dist=1.0 (start), kappa = 1.5
    # At normalized_dist=0.0 (end), kappa = 0.2
    kappa = kappa_low + (kappa_high - kappa_low) * normalized_dist
    
    # LCB Score: mu - kappa * sigma
    # We want to MINIMIZE f(x).
    # LCB is a lower bound on f(x). We want the lowest LCB.
    # Acquisition score (higher = better) = -LCB = kappa * sigma - mu
    raw_score = kappa * sigma - mu
    
    # Cost Normalization
    # Prefer cheaper evaluations.
    # Using power 1.0 is standard and avoids over-penalizing cost which might
    # skew the search away from promising but slightly more expensive regions.
    final_scores = raw_score / cost_safe
    
    # Handle non-finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, -1e10)
    
    return final_scores
# EVOLVE-BLOCK-END
