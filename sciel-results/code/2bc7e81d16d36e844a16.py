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
    # The global minimum of cosine8 on [-1, 1]^8 is -0.8.
    # We assume the initial uniform points yield values roughly around 0.0 (mean of cosine).
    # We define a "progress" metric based on how close f_best is to the global min.
    # progress = 1.0 means we are at the global min (exploit hard).
    # progress = 0.0 means we are far from the min (explore hard).
    
    f_min_global = -0.8
    f_initial_estimate = 0.0  # Typical initial value for cosine8
    
    # Calculate raw progress
    # If f_best is 0.0, progress = 0.0 (start of search) -> High Kappa
    # If f_best is -0.8, progress = 1.0 (end of search) -> Low Kappa
    if f_initial_estimate == f_min_global:
        progress = 0.0
    else:
        progress = (f_best - f_min_global) / (f_initial_estimate - f_min_global)
    
    # Clip progress to [0, 1] to handle cases where f_best might be > 0 or < -0.8
    progress = np.clip(progress, 0.0, 1.0)
    
    # Define Kappa range
    # High exploration in early stages (high progress value in our inverted logic? 
    # Wait, let's re-evaluate the mapping).
    # 
    # Let's use a direct mapping:
    # Distance from optimum = f_best - (-0.8) = f_best + 0.8
    # If f_best is 0.0, dist = 0.8.
    # If f_best is -0.8, dist = 0.0.
    # 
    # We want HIGH kappa when dist is LARGE (early).
    # We want LOW kappa when dist is SMALL (late).
    
    # Normalize distance to [0, 1]
    # Max expected distance is roughly 0.8 (from 0.0 to -0.8)
    max_dist = 0.8 
    dist = np.maximum(0.0, f_best - f_min_global)
    normalized_dist = np.clip(dist / max_dist, 0.0, 1.0)
    
    # Kappa interpolation
    # Use a quadratic decay for exploration to maintain high exploration
    # early on and switch to exploitation more aggressively as we approach
    # the optimum, reducing the risk of getting stuck in local minima
    # while still exploiting the mean.
    kappa_low = 0.4
    kappa_high = 2.2
    
    # Quadratic mapping: normalized_dist^2
    # At dist=1, kappa=2.2. At dist=0.5, kappa=0.4+2.2*0.25=0.95. At dist=0, kappa=0.4.
    kappa = kappa_low + (kappa_high - kappa_low) * (normalized_dist ** 2)
    
    # LCB Score: mu - kappa * sigma
    # We want to MINIMIZE this value.
    # Acquisition score should be HIGH when LCB is LOW.
    # So, acquisition_raw = -(mu - kappa * sigma) = kappa * sigma - mu
    
    lcb_value = mu - kappa * sigma
    raw_score = -lcb_value  # This is kappa * sigma - mu
    
    # Cost Normalization
    # Divide by cost to prefer cheaper evaluations.
    # Use a slightly super-linear penalty (power 1.1) to more strongly
    # discourage high-cost evaluations when the acquisition value is similar,
    # effectively conserving budget for more evaluations.
    final_scores = raw_score / (cost_safe ** 1.1)
    
    # Handle non-finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, -1e10)
    
    return final_scores
# EVOLVE-BLOCK-END
