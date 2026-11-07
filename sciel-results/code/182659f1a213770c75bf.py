"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Thompson Sampling approach:
    # Sample from the predictive posterior for each candidate
    # Score is the negative of the sample, so higher scores correspond to lower objective values
    # This naturally balances exploration and exploitation
    
    # To make Thompson Sampling effective, we want to sample in a way that
    # encourages exploration of uncertain regions but also exploits known good regions
    
    # A simple approach: sample directly from N(mu, sigma^2)
    # But to ensure we get a good balance, we can use a modified approach
    
    # For each candidate, draw a sample from N(mu_i, sigma_i^2)
    # The score is -sample, so we want to maximize -sample = minimize sample
    
    # However, naive Thompson Sampling can be too exploratory or too exploitative
    # We can adapt by using a temperature parameter that scales the uncertainty
    
    # Adaptive temperature: reduce exploration as we converge
    # When sigma is large (early on), explore more; when sigma is small (later), exploit more
    
    # Compute the mean uncertainty to adapt the sampling
    mean_sigma = np.mean(sigma)
    
    # If mean uncertainty is very small, we're in exploitation mode
    # If mean uncertainty is large, we're in exploration mode
    # Scale the sampling uncertainty accordingly
    
    # Use a soft adaptation: effective_sigma = sigma * (1 + alpha * (1 / (1 + mean_sigma)))
    # where alpha controls how much we boost exploration when uncertainty is low
    # But actually, we want the opposite: when uncertainty is low, we should explore less
    
    # Simpler: just use the natural Thompson Sampling with the given sigma
    # The GP already provides calibrated uncertainty, so we trust it
    
    # Draw samples from the predictive posterior
    # To ensure reproducibility within a single call but different across calls,
    # we use a random generator seeded by the current state
    
    # For efficiency and to avoid issues with random state, we use numpy's random
    # We draw 2000 samples, one for each candidate
    
    # The key insight: Thompson Sampling samples s_i ~ N(mu_i, sigma_i^2)
    # and we return -s_i as the score
    
    # But we need to be careful: if sigma is 0, we should return -mu
    # If sigma is very small, the sample is close to mu
    
    # Let's draw the samples
    rng = np.random.default_rng()
    
    # For points with very small sigma, the sample is essentially mu
    # For points with large sigma, the sample has more variance
    
    # Draw samples: s_i = mu_i + sigma_i * z_i, where z_i ~ N(0, 1)
    z = rng.normal(0, 1, size=mu.shape)
    samples = mu + sigma * z
    
    # Score is the negative of the sample (we want to minimize the objective)
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
