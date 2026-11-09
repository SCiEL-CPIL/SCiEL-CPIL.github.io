import numpy as np
import time

# Explicit deadline guard variables
_START_TIME = time.time()
# The evaluator kills after 330s. We set a safety margin.
# Since acquisition is called 25 times, and each call is fast,
# the total time is dominated by the GP fitting/evaluation in the evaluator loop.
# However, we must comply with the directive to have a guard.
_DEADLINE = _START_TIME + 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Thompson Sampling.
    
    Parameters:
    mu (np.ndarray): Posterior means, shape (N,)
    sigma (np.ndarray): Posterior standard deviations, shape (N,)
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate points 
                that should be evaluated next.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback to exploitation (minimize mean) if we are running out of time
        return -mu

    # Thompson Sampling Strategy:
    # 1. Draw a sample y_i from the GP posterior N(mu_i, sigma_i^2) for each candidate x_i.
    # 2. The point with the minimum sample y_min is the one we want to evaluate.
    # 3. The evaluator selects the point with the MAXIMUM acquisition score.
    # 4. Therefore, we return -y_i as the score.
    
    # Generate standard normal random numbers for each candidate
    # Using np.random.randn is efficient and vectorized
    z = np.random.randn(len(mu))
    
    # Compute the sampled values
    # sigma is the standard deviation
    sample = mu + sigma * z
    
    # Invert the sign so that the minimum sample value corresponds to the maximum score
    scores = -sample
    
    # Ensure all scores are finite. 
    # If any value is NaN or Inf, replace it with a very low score (-1e10) 
    # to deprioritize it, although GP outputs should be finite.
    scores[~np.isfinite(scores)] = -1e10
    
    return scores
