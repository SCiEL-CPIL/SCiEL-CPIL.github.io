import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling
    # We want to minimize the function value.
    # We sample from the posterior predictive distribution.
    # y ~ N(mu, sigma^2)
    
    # Generate standard normal random numbers
    # Shape of mu is (2000,)
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from posterior
    y = mu + sigma * z
    
    # We want to select the point that minimizes y, penalized by cost.
    # Score should be higher for lower y and lower cost.
    # Using -y / cost achieves this.
    # To avoid numerical issues if cost is 0 (it's not, min 1), we can just divide.
    
    scores = -y / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
