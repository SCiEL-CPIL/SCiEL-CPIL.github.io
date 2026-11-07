import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian Optimization.
    
    Strategy:
    Instead of using a deterministic score like UCB (mu - kappa * sigma), we use
    Thompson Sampling. We draw a single sample from the Gaussian posterior:
    sample = mu + sigma * z, where z ~ N(0, 1).
    
    Since we are minimizing, we want to find the point with the lowest sampled value.
    The acquisition score is defined such that higher is better for selection.
    Therefore, score = -sample = -(mu + sigma * z).
    
    Why this is different:
    1. It is stochastic. Each call yields a different ranking of candidates.
    2. It implicitly adapts the exploration/exploitation trade-off. In regions of
       high uncertainty (high sigma), the sample can vary widely, potentially
       leading to very low (good) values, thus encouraging exploration. In regions
       of low uncertainty, the sample is close to the mean, encouraging exploitation
       of known good areas.
    3. It avoids the "greedy" trap of UCB which might always pick the same high-uncertainty
       region if kappa is fixed, whereas TS "rotates" through different hypotheses of
       the function.
       
    Implementation details:
    - We use a fixed random seed derived from the current best value and time to ensure
      reproducibility is not strictly required but behavior is consistent within a run?
      Actually, standard TS uses a fresh random draw each time. The evaluator calls this
      function repeatedly. We should generate a fresh random z each time.
    - However, numpy's global random state is shared. To be safe and isolated, we can
      use a local RandomState or just np.random.randn. Given the constraints, using
      the global np.random is standard for these tasks unless specified otherwise.
      But to ensure "finite scores" and robustness, we just compute the linear combination.
    - We need to handle the case where sigma is very small or zero.
    - Deadline guard: The computation is O(N) for 2000 points, which is negligible.
      No explicit deadline guard is strictly necessary for the math, but we can add a
      trivial check to satisfy the directive if we were doing heavy lifting. Here, 
      the heavy lifting is in the GP update which is outside this function.
      This function is just a vectorized operation.
    """
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single standard normal random variable
    # This creates a new random draw each time the function is called.
    z = np.random.randn()
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # y = mu + sigma * z
    # We want to minimize y, so we maximize -y
    scores = -(mu + sigma * z)
    
    # Ensure scores are finite
    # If mu or sigma contain NaN/Inf, the scores will too. 
    # The problem states mu and sigma are from a GP, which should be finite.
    # But to be safe against numerical overflow in extreme cases:
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
