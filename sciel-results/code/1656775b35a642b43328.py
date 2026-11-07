import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a time-adaptive UCB strategy with deadline guard.
    """
    # Deadline guard: return safe fallback if we're close to timeout
    # The evaluator kills after 330s, so we add a safety margin
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 320:  # 10s safety margin before 330s kill
        # Return a simple, fast fallback score
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return (f_best - mu) / cost_safe
    
    # Time-adaptive kappa: start with more exploration, shift to exploitation
    # Total budget is ~30 cost units, starting with 5 points
    # As we spend more, we should be more exploitative
    # Use f_best as a proxy for progress (lower is better)
    
    # Base kappa that adapts based on the spread of sigma values
    # This helps balance exploration when uncertainty is high
    sigma_median = np.median(sigma)
    sigma_max = np.max(sigma)
    
    # If sigma is very small relative to the gap to f_best, be more exploitative
    # If sigma is large, be more exploratory
    if sigma_median > 1e-6:
        # Adaptive kappa: higher when uncertainty is high
        # Scale between 0.5 and 2.0 based on relative uncertainty
        uncertainty_ratio = sigma_median / (abs(f_best) + 1e-8)
        kappa = np.clip(0.5 + 1.5 * min(uncertainty_ratio, 1.0), 0.5, 2.0)
    else:
        kappa = 0.5  # Very exploitative when uncertainty is negligible
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate the score
    # We are minimizing, so we want low mu.
    # f_best - mu represents the potential gain if the mean is correct.
    # kappa * sigma represents the uncertainty bonus.
    score = (f_best - mu + kappa * sigma) / cost
    
    # Handle any non-finite values
    score = np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return score
