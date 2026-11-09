"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Strategy:
    - Sample one value from the posterior Gaussian N(mu, sigma^2) for each candidate.
    - For minimization, we want the candidate with the lowest sampled value.
    - The interface requires returning scores where HIGHER is better.
    - Therefore, we return the negative of the sampled value: score = -sample.
    
    This approach naturally balances exploration and exploitation:
    - Where sigma is high, the sample has a higher chance of being very low (exploration).
    - Where mu is low, the sample is likely to be low (exploitation).
    - No manual tuning of exploration constants (like kappa in UCB) is required.
    
    Deadline guard:
    - The operation is O(N) vectorized numpy random sampling and arithmetic.
    - For N=2000, this is extremely fast (< 1ms).
    - We include a basic finite check to ensure validity.
    """
    start_time = time.time()
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 0.0)
    
    # Thompson Sampling: Draw a sample from N(mu, sigma^2) for each candidate
    # Using standard normal random numbers
    # To ensure reproducibility within a run is not strictly required by the contract,
    # but standard random state is fine.
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z
    
    # For minimization, we want the minimum sample.
    # Score interface: Higher is better.
    # So we return -samples.
    score = -samples
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check (should never trigger for N=2000, but good practice)
    if time.time() - start_time > 250.0:
        # Fallback to simple exploitation if something went wrong (unlikely)
        return -mu
    
    return score
# EVOLVE-BLOCK-END
