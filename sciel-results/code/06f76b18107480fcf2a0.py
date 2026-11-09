import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Use Thompson Sampling: sample from the posterior and evaluate the negative cost-normalized value
    # This is a structural change from deterministic UCB/EI to a stochastic method.
    # y ~ N(mu, sigma^2)
    # We want to minimize y, so we maximize -y / cost
    
    # Generate standard normal samples
    # mu and sigma have shape (2000,)
    z = np.random.randn(len(mu))
    
    # Sample from the GP posterior
    y = mu + sigma * z
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Score: higher is better. We want low y and low cost.
    score = -y / safe_cost
    
    return score
