import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Log-Probability of Improvement (LogPI).
    
    Structural change: 
    - Uses LogPI instead of EI for better scale properties and sharper gradients.
    - Normalizes by cost to directly optimize cost-effectiveness.
    - Adds a decaying sigma-based exploration term to handle the multimodal nature 
      of cosine8 in 8D, where early exploration is critical.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 325.0:
        # Fallback: simple heuristic if time is running out
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1.0)
    
    cost_safe = np.maximum(cost, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)

    # Calculate z-score: (f_best - mu) / sigma
    # Higher z means higher probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical overflow in exp/log
    z_clipped = np.clip(z, -10.0, 50.0)
    
    # Compute Log-Probability of Improvement (LogPI)
    # P(y < f_best) = Phi(z)
    # We want to maximize log(Phi(z))
    # Using scipy.special.ndtr for accuracy, but we need log.
    # scipy.special.ndtr(z) gives Phi(z).
    # To compute log(Phi(z)) safely:
    # If z is large positive, log(Phi(z)) is close to 0.
    # If z is negative, Phi(z) is small, log is large negative.
    # We can use scipy.special.log_ndtr if available, otherwise compute carefully.
    
    # Since standard library + numpy/scipy is allowed, let's use scipy.special.log_ndtr
    try:
        from scipy.special import log_ndtr
        log_pi = log_ndtr(z_clipped)
    except ImportError:
        # Fallback calculation if log_ndtr is not available
        # Approximation for log(Phi(z))
        # For z > 0: log(Phi(z)) ~ -exp(-0.5*z^2) * sqrt(2/pi) / z (very small negative)
        # For z < 0: log(Phi(z)) ~ log(Phi(-z)) - ... actually just use norm.cdf
        from scipy.stats import norm
        cdf_vals = norm.cdf(z_clipped)
        # Avoid log(0)
        cdf_vals = np.maximum(cdf_vals, 1e-300)
        log_pi = np.log(cdf_vals)
        
    # Cost-aware LogPI
    # We divide by cost to prefer cheap points
    cost_log_pi = log_pi / cost_safe

    # Exploration Term:
    # Encourage exploring high sigma regions, but decay this influence as we get closer to the optimum.
    # f_best starts high (e.g., 0) and goes down to -0.8.
    # Let's define a progress metric.
    # Range of cosine8 is approx [-0.8, 0.8].
    # progress = (f_best - min_val) / (max_val - min_val)
    # When f_best is high, progress is high (need exploration).
    # When f_best is low, progress is low (exploit).
    
    min_val = -0.8
    max_val = 0.8
    range_val = max_val - min_val
    
    # Clamp f_best to known range for progress calculation
    f_best_clipped = np.clip(f_best, min_val, max_val)
    progress = (f_best_clipped - min_val) / range_val
    # progress is 1.0 when f_best is at max (start), 0.0 when f_best is at min (end)
    
    # Exploration weight decays linearly with progress
    # Start with significant exploration, end with almost none
    exploration_weight = 0.5 * progress
    
    # Exploration score: sigma / cost
    # Normalize sigma to be comparable to log_pi? 
    # log_pi is roughly in [-inf, 0]. sigma is [0, ...].
    # Let's just add a scaled sigma term.
    # To prevent sigma from dominating when cost is small, we normalize by a typical sigma scale.
    # However, simple sigma/cost is a common UCB variant.
    # Let's blend it into the score.
    exploration_score = exploration_weight * (sigma_safe / cost_safe)
    
    # Combine
    # We want to maximize.
    # log_pi is <= 0.
    # exploration_score is >= 0.
    # This creates a bias towards exploration in early stages.
    
    final_score = cost_log_pi + exploration_score
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
