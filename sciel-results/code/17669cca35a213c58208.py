import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from the posterior
    # For minimization, we want to sample values that are likely to be low
    # Sample: x ~ mu - sigma * Z, where Z ~ N(0,1)
    # This gives us a random draw from the posterior predictive distribution
    # We want to maximize the score, so we negate the sample
    rng = np.random.default_rng()
    samples = mu - sigma * rng.standard_normal(len(mu))
    
    # The score is the negative of the sample (higher score = lower predicted value)
    scores = -samples
    
    return scores
