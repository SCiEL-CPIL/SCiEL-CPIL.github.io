import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Use a fixed seed for the local RNG to ensure reproducibility
    # The 'randomness' is actually a deterministic set of multipliers
    # This avoids global RNG state issues.
    
    # Generate standard normal samples
    # We use a fixed seed, so the sequence of random numbers is always the same.
    # This means the "exploration" is deterministic but varied across the 2000 points.
    rng = np.random.default_rng(12345)
    z = rng.standard_normal(mu.shape)
    
    # Add a small epsilon to sigma to prevent division by zero or extreme values
    # though we are multiplying here, so it's fine.
    # However, if sigma is 0, the sample is exactly mu.
    
    samples = mu + z * (sigma + 1e-9)
    
    # We want to minimize the objective function.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we return the negative of the sampled value.
    return -samples
