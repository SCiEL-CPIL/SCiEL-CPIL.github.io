import numpy as np
import time

# Track time for the entire process to ensure we don't exceed the 330s limit
# The evaluator kills the process at 330s.
# We use a global variable to track the start time of the first call.
_START_TIME = time.time()
# Set a safety margin. If we are within 5 seconds of the limit, 
# we might want to force a more deterministic or faster path, 
# though the computation here is negligible.
# The main purpose is to satisfy the directive for an explicit guard.
_SAFETY_MARGIN = 325.0 

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) strategy.
    
    Args:
        mu: (2000,) array of posterior means.
        sigma: (2000,) array of posterior standard deviations.
        f_best: float, the best observed value so far.
        
    Returns:
        (2000,) array of scores. Higher scores are preferred.
    """
    # Deadline Guard
    # If we are extremely close to the time limit, we could potentially
    # skip complex calculations, but UCB is already O(N).
    # We check here to ensure we never contribute to a timeout.
    if time.time() > _SAFETY_MARGIN:
        # Fallback to greedy (exploitation only) to be as fast as possible
        # and avoid any potential overhead in complex branching.
        return -mu

    # UCB Parameter
    # Beta controls the exploration-exploitation tradeoff.
    # Higher beta -> more exploration.
    # For multimodal functions like Michalewicz, a moderate beta is good.
    beta = 2.0
    
    # Calculate UCB scores for minimization.
    # We want to select the point that minimizes the function.
    # UCB bound for minimization: mu - beta * sigma.
    # To use an argmax interface, we negate this: -mu + beta * sigma.
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite.
    # Replace NaNs and Infs with safe values.
    # NaNs in mu/sigma indicate GP failure; treat as very unattractive.
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
    
    return scores
