import numpy as np
from scipy.stats import norm
import time

# Module-level state to track the number of evaluations
_eval_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using GP-UCB with time-varying beta.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at each candidate point.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    global _eval_count
    _eval_count += 1
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-10)  # avoid division by zero
    
    # GP-UCB for minimization:
    # We want to minimize f, so we use:
    # UCB(x) = -mu(x) + sqrt(beta_t) * sigma(x)
    # This encourages points where the mean is low (good) and uncertainty is high (exploration).
    #
    # Time-varying beta: starts high for exploration, decreases for exploitation.
    # Using beta_t = c * sqrt(log(t + 1)) with a tuned constant c.
    # For 30 evaluations total, we want beta to be moderate.
    # Let's use beta_t = sqrt(log(t + 1) + 1) * scale
    # where scale is chosen to balance exploration and exploitation.
    
    t = _eval_count
    # Beta schedule: start with more exploration, reduce over time
    # beta_t = sqrt(log(t + 1) + 1) gives:
    # t=1: sqrt(log(2)+1) ≈ sqrt(1.693) ≈ 1.30
    # t=5: sqrt(log(6)+1) ≈ sqrt(2.79) ≈ 1.67
    # t=10: sqrt(log(11)+1) ≈ sqrt(3.40) ≈ 1.84
    # t=20: sqrt(log(21)+1) ≈ sqrt(4.04) ≈ 2.01
    # t=30: sqrt(log(31)+1) ≈ sqrt(4.43) ≈ 2.10
    
    # This increases slowly, which is good. But we want to start with more exploration.
    # Let's use a different schedule that starts higher and decreases.
    # Or we can use a fixed beta that works well for this problem size.
    
    # For 30 evaluations, a moderate beta around 1.0-2.0 should work.
    # Let's use beta_t = 1.5 * sqrt(log(t + 1))
    # t=1: 1.5 * sqrt(0.693) ≈ 1.24
    # t=5: 1.5 * sqrt(1.79) ≈ 2.00
    # t=10: 1.5 * sqrt(2.40) ≈ 2.32
    # t=20: 1.5 * sqrt(3.04) ≈ 2.62
    # t=30: 1.5 * sqrt(3.43) ≈ 2.77
    
    # Actually, for minimization with only 30 evals, we should be more exploratory early.
    # Let's try beta_t = 2.0 * sqrt(log(t + 1) + 0.5)
    # t=1: 2.0 * sqrt(1.193) ≈ 2.18
    # t=5: 2.0 * sqrt(2.29) ≈ 3.03
    # t=10: 2.0 * sqrt(2.90) ≈ 3.40
    # t=20: 2.0 * sqrt(3.54) ≈ 3.76
    # t=30: 2.0 * sqrt(3.93) ≈ 3.96
    
    # This might be too exploratory. Let's use a simpler approach:
    # beta_t = c / (1 + t/10) to decrease exploration over time
    # Or just use a fixed moderate beta.
    
    # Given the small number of evaluations (30), let's use:
    # beta_t = 1.5 + 0.5 * (1 / (1 + t/5))
    # t=1: 1.5 + 0.5*(1/1.2) ≈ 1.92
    # t=5: 1.5 + 0.5*(1/2) = 1.75
    # t=10: 1.5 + 0.5*(1/3) ≈ 1.67
    # t=20: 1.5 + 0.5*(1/5) = 1.60
    # t=30: 1.5 + 0.5*(1/7) ≈ 1.57
    
    # This starts with moderate exploration and settles to exploitation.
    beta_t = 1.5 + 0.5 * (1.0 / (1.0 + t / 5.0))
    
    # GP-UCB for minimization
    scores = -mu + beta_t * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
