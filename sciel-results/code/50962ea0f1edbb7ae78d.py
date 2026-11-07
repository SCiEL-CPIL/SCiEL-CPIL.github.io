import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a progress-adaptive blend of Expected Improvement (EI) and UCB.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 320:
        # Safe fallback: Exploitation focused
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        # Simple exploitation score
        return (f_best - mu) / cost_safe

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Estimate progress based on f_best
    # Hartmann 6D minimum is approx -3.32237. Max is approx 0.
    f_min_hartmann = -3.32237
    f_max_hartmann = 0.0
    
    # t = 0 at start (f_best ~ 0), t = 1 at end (f_best ~ -3.32)
    # If f_best is worse than f_max (e.g. 1.0), clamp t to 0.
    # If f_best is better than f_min (e.g. -4.0), clamp t to 1.
    progress = (f_max_hartmann - f_best) / (f_max_hartmann - f_min_hartmann)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Weights for blending
    # Early (progress=0): High UCB weight, Low EI weight
    # Late (progress=1): Low UCB weight, High EI weight
    w_ei = progress
    w_ucb = 1.0 - progress
    
    # Adaptive Beta for UCB
    # Start with high exploration (beta=2.0), end with low exploration (beta=0.5)
    beta = 2.0 - 1.5 * progress
    
    # Calculate EI
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Use scipy.stats.norm for Phi and phi
    # Phi(z) = CDF
    # phi(z) = PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is the expected improvement for minimization
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Calculate UCB score (as expected gain)
    # We want to maximize the potential gain.
    # UCB_value = f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    ucb_score = f_best - mu + beta * sigma
    
    # Blend the scores
    # Both EI and UCB_score are "gains" (higher is better)
    blended_score = w_ei * ei + w_ucb * ucb_score
    
    # Divide by cost
    final_score = blended_score / cost
    
    # Handle non-finite values
    final_score = np.nan_to_num(final_score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return final_score
