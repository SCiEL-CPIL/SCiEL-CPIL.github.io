import numpy as np
import time

def acquisition(mu, sigma, f_best):
        start = time.time()
        
        # Thompson Sampling
        # We sample from the posterior predictive distribution
        # N(mu, sigma^2)
        # We want to minimize the function, so we want the sample with the lowest value.
        # The acquisition score is the negative of the sample.
        
        # Add a small amount of exploration noise? No, TS is already stochastic.
        
        # Ensure sigma is positive
        sigma_safe = np.maximum(sigma, 1e-9)
        
        samples = np.random.normal(mu, sigma_safe)
        
        # Check time
        if time.time() - start > 290:
            # Fallback to EI if TS is slow (it won't be, but just in case)
            z = (f_best - mu) / sigma_safe
            from scipy.stats import norm
            return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
            
        return -samples
