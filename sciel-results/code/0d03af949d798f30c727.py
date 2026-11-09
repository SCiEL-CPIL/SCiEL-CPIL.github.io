import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = 300 - 5  # 5 seconds safety margin
    
    # Ensure sigma is not zero to avoid division by zero
    # Using a small epsilon
    eps = 1e-9
    
    # The directive suggests: (f_best - mu) / (sigma + epsilon)
    # This is the z-score of the improvement, equivalent to PI (monotonic)
    # We maximize this score.
    
    # Check deadline (though computation is very fast)
    if time.time() > deadline:
        # Fallback to a simple heuristic if time runs out (unlikely)
        # Return scores based on low mu
        return -mu
        
    scores = (f_best - mu) / (sigma + eps)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
