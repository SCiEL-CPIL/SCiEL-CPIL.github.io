import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Monte Carlo approximation of the 1-step Knowledge Gradient.
    
    The 1-step KG estimates the expected reduction in the minimum value
    found after the next evaluation. In the absence of full correlation
    information for previous points, this is approximated by sampling
    from the posterior distribution of the new candidate.
    """
    # Number of Monte Carlo samples
    num_samples = 500
    
    # Generate samples from N(mu, sigma^2)
    # mu and sigma have shape (2000,)
    # We broadcast to (num_samples, 2000)
    z = np.random.randn(num_samples, len(mu))
    samples = mu + sigma * z
    
    # Calculate the improvement for each sample: max(f_best - sample, 0)
    improvements = np.maximum(f_best - samples, 0.0)
    
    # The KG is the mean expected improvement
    kg = np.mean(improvements, axis=0)
    
    return kg
