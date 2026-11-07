import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = 325.0 # 5 seconds margin from 330s
    
    # Thompson Sampling
    # Use a local RNG to ensure independence from global state
    rng = np.random.default_rng()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Sample from the posterior
    z = rng.standard_normal(mu.shape)
    
    # y ~ N(mu, sigma^2)
    y = mu + sigma * z
    
    # We want to minimize y.
    # Return negative y so that the minimum y has the maximum score.
    scores = -y
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline check (though TS is fast, this satisfies the directive)
    if time.time() > deadline:
         # Return the best found so far? 
         # TS doesn't have a "loop" inside, it's a single vector op.
         # So this check is just a formality.
         pass
         
    return scores
