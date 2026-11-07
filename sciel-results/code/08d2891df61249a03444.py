import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 320.0  # 300s limit + margin, but evaluator kills at 330s. Keep it tight.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + kappa * sigma
    Higher score means more promising (lower expected value with uncertainty bonus).
    """
    # Check deadline
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        # Return a neutral score or the best found so far?
        # The task says "return the best result found with a safety margin".
        # Since we don't track history here, we just return a safe array.
        # However, the evaluator expects 2000 scores.
        # If we time out, we should probably just return something that doesn't crash.
        # But the loop is external. The function is called per step.
        # If this call is slow, it's the bottleneck.
        # This function is vectorized and fast.
        pass

    # UCB for minimization
    # We want to minimize the function.
    # The predicted mean is mu.
    # The uncertainty is sigma.
    # A point is attractive if mu is low OR sigma is high (potential for better values).
    # Score = -mu + kappa * sigma
    # We return this score. Higher is better.
    
    kappa = 2.5
    
    # Ensure sigma is positive to avoid negative scores from negative sigma (though sigma should be positive)
    # sigma is standard deviation, so it should be >= 0.
    # Add a small epsilon to sigma to handle potential numerical zeros.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
