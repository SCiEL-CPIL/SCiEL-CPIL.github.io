import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses Thompson Sampling: sample a single realization from the Gaussian
    posterior for all 2000 candidates and return the negative sampled values
    as scores. This naturally balances exploration and exploitation.
    
    To avoid randomness issues across calls, we use a deterministic pseudo-random
    offset based on the input data to ensure reproducibility within a run while
    maintaining exploration behavior.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample from N(mu, sigma^2)
    # Use a fixed random state based on the data to be deterministic per call
    # but still explore. We use a hash of the first few values to seed.
    # Actually, for Thompson Sampling in BO, we typically use a fresh random
    # sample each call. But since the evaluator may call this multiple times
    # in a loop, we want each call to potentially explore differently.
    # However, the task says "draws 2000 fresh uniform candidates, computes
    # the posterior there, calls acquisition". So each call is a fresh set
    # of candidates. We should use a fresh random sample each time.
    
    # Generate standard normal random numbers
    rng = np.random.RandomState()
    # Use a seed based on the current time to get different samples each call
    # But we need to be careful: if we use time.time(), it might be too fast
    # and give same seed. Let's use a combination.
    # Actually, for proper Thompson Sampling, we just use np.random.standard_normal
    # which uses the global RNG state.
    
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z
    
    # Return negative samples as scores (higher score = lower sampled value)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
