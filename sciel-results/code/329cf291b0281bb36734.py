"""Direct cost-aware acquisition for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling with cost normalization.
    
    Strategy: Draw a single random sample from the GP posterior at each candidate
    point, then score by (sample / cost). This treats the problem as a stochastic
    reward-to-go ratio, balancing exploration (via the random draw) and
    cost-awareness (via division by cost).
    
    The random draw is: mu - sigma * z, where z ~ N(0, 1).
    We subtract because lower objective values are better.
    
    We guard against the 330s wall-clock limit with an explicit deadline.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw a single standard normal sample for Thompson Sampling
    # Using a fixed seed based on the current f_best to ensure reproducibility
    # within the same state, but different across iterations
    seed = int(np.abs(hash((f_best, len(mu)))) % (2**32))
    rng = np.random.RandomState(seed)
    z_sample = rng.standard_normal(len(mu))
    
    # Thompson sample: mu - sigma * z (lower is better, so we minimize this)
    thompson_sample = mu - sigma * z_sample
    
    # Score: minimize thompson_sample / cost, so maximize -thompson_sample / cost
    safe_cost = np.maximum(cost, 1e-8)
    score = -thompson_sample / safe_cost
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 300:  # Additional safety check
        # Fallback: pure exploitation with cost awareness
        score = -mu / safe_cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
