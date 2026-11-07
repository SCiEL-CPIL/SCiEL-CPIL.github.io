"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling:
    # Sample from the posterior distribution N(mu, sigma)
    # The acquisition score is the negative of the sample, so higher score = lower value.
    # This encourages exploration in high-uncertainty regions naturally.
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Avoid zero variance issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples from the posterior
    # Using a fixed random state is not possible here as we don't have state, 
    # but numpy's global random state is fine.
    # We sample from N(mu, sigma)
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # We want to minimize the objective, so we return -sample as the score (higher is better)
    # However, the problem asks for "finite scores (higher = evaluate next)".
    # If we return -sample, the point with the lowest predicted value gets the highest score.
    # This is the core of Thompson Sampling for minimization.
    
    # To add a slight cost-awareness or exploration bonus that isn't pure random noise,
    # we can blend with a small amount of uncertainty or just stick to pure TS.
    # Pure TS is often robust. Let's stick to pure TS first as it's a structural change.
    
    # One potential issue: if sigma is very small, it's exploitation. If large, exploration.
    # This is exactly what we want.
    
    scores = -samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
