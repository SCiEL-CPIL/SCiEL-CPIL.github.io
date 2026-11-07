import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Draw a sample from the posterior distribution N(mu, sigma^2)
    # We want to minimize the objective, so we pick the candidate with the lowest sampled value.
    # The function must return scores where HIGHER is better.
    # So we return the negative of the sampled value.
    
    # Use a fixed random seed? No, the evaluator might call this multiple times.
    # Using global random state is standard.
    
    z = np.random.randn(mu.shape[0])
    samples = mu + sigma * z
    
    # Return negative samples so that lower objective values correspond to higher scores
    return -samples
