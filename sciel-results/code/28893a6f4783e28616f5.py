# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to persist across calls within the same seed run
_acq_state = {
    'rng': np.random.default_rng()
}

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    deadline = t_start + 295.0  # Safety margin for 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is safe for division/usage
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Strategy: Thompson Sampling
    # Instead of computing a deterministic acquisition score like EI or UCB,
    # we sample a single value from the posterior distribution at each candidate.
    # The candidate with the lowest sampled value (since we minimize) is selected.
    # This naturally balances exploration (high sigma leads to high variance in samples)
    # and exploitation (low mu leads to lower samples).
    
    # To ensure the function returns scores where HIGHER is better (as per contract: 
    # "higher meaning evaluate next"), we will return the negative of the sampled values.
    # Alternatively, we can sample and then rank them. 
    # The contract says: "Return 2000 finite scores, higher meaning evaluate next".
    
    # Draw one sample from N(mu, sigma) for each of the 2000 candidates
    # We use a single random draw for all points to ensure a consistent "view" of the function
    # This is the standard Thompson Sampling approach for BO.
    
    # Get a single random normal variate for the "noise" component? 
    # No, each point has its own sigma. We draw 2000 independent samples.
    # sample_i ~ N(mu_i, sigma_i)
    
    # To be efficient and reproducible within the call:
    rng = _acq_state['rng']
    z = rng.standard_normal(n)
    samples = mu + sigma_safe * z
    
    # We want to MINIMIZE the objective.
    # So the "best" sample is the one with the lowest value.
    # The contract requires HIGHER scores to mean "evaluate next".
    # Therefore, we return -samples.
    
    scores = -samples
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Time check (Thompson sampling is very fast, O(N))
    if time.time() > deadline:
        pass
        
    return scores

# EVOLVE-BLOCK-END
