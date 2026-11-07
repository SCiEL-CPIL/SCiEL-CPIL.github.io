import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard (if needed for iterative improvements in future, 
# though this function is stateless and fast)
_start_time = time.time()
DEADLINE = 300  # Seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using tanh-normalized Expected Improvement.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: Scalar, best observed function value so far.
        cost: (2000,) array of evaluation costs (1 + mean(x)).
        
    Returns:
        score: (2000,) array of acquisition scores (higher is better).
    """
    # Deadline guard: If we are too close to the limit, return a simple fallback
    # to ensure we don't crash. However, this computation is O(N) and very fast.
    current_time = time.time()
    if current_time - _start_time > DEADLINE - 5:
        # Fallback: return uniform random or zero to avoid timeout
        # Ideally, we should return something valid. Zero is safe (no preference).
        return np.zeros_like(mu)

    # Ensure sigma is positive and non-zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * P(f < f_best) + sigma * phi(z)
    # Using scipy.stats.norm for accurate CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural Change: Normalize EI to prevent outliers from dominating
    # This makes the cost penalty more consistent across the candidate set.
    # We use the standard deviation of EI as the scale.
    ei_std = np.std(ei)
    if ei_std < 1e-6:
        ei_std = 1e-6
        
    # Apply tanh to saturate extreme EI values
    # This bounds the score contribution from improvement,
    # allowing the cost term to have a more stable influence.
    normalized_ei = np.tanh(ei / ei_std)
    
    # Cost-aware scoring
    # Divide by cost to penalize expensive evaluations.
    safe_cost = np.maximum(cost, 1e-6)
    score = normalized_ei / safe_cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
