import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Structural change: Thompson Sampling
        # Generate a seed based on the current state to ensure stochasticity
        # that evolves with the GP updates.
        
        # Use f_best and the state of the GP (mu, sigma) to create a unique seed
        # This ensures that if f_best is the same, but the GP has changed (new points added),
        # the random draw is different.
        
        # Hash components
        seed_val = int((abs(f_best) + np.mean(mu) + np.mean(sigma)) * 1e6) % (2**32)
        
        rng = np.random.RandomState(seed_val)
        
        # Draw standard normal samples
        z = rng.randn(len(mu))
        
        # Sample from the posterior predictive distribution
        # f_sample = mu + sigma * z
        f_sample = mu + sigma * z
        
        # We want to minimize f, so we maximize -f_sample
        # Return scores such that higher is better
        return -f_sample
