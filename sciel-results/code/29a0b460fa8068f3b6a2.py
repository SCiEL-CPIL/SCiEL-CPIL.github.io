import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling for minimization
    # We draw a sample from the posterior predictive distribution
    # and select the point with the lowest sampled value.
    # Since the evaluator maximizes the score, we return the negative of the sample.
    
    # Handle numerical stability for standard deviation
    eps = 1e-8
    safe_sigma = np.maximum(sigma, eps)
    
    # Draw samples from N(mu, sigma)
    samples = np.random.normal(mu, safe_sigma)
    
    # We want to minimize the objective, so we prefer low samples.
    # The acquisition function should return higher scores for preferred points.
    # Therefore, we negate the samples.
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
