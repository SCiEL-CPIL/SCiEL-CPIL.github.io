import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a sample from the posterior and score by -sample.
    # This provides a natural balance between exploration and exploitation
    # by leveraging the full posterior distribution.
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single random sample from the posterior at each candidate point
    # theta = mu + sigma * Z, where Z ~ N(0, 1)
    Z = np.random.standard_normal(mu.shape[0])
    theta = mu + sigma_safe * Z
    
    # Score by negative of the sample (higher score = lower predicted value)
    scores = -theta
    
    return scores
