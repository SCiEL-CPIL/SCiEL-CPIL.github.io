# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score using Cost-Aware Thompson Sampling.
    Returns a (2000,) array of finite scores (higher = more preferred).
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return inverse cost
        return 1.0 / np.maximum(cost, 1e-10)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, 1e-10)
    
    n_candidates = len(mu)
    
    # Number of samples for Thompson Sampling
    # 20 samples is a good balance between computational cost and estimation accuracy for N=2000
    n_samples = 20
    
    # Initialize counts
    counts = np.zeros(n_candidates, dtype=np.float64)
    
    # Generate samples from the posterior
    # mu and sigma are (N,)
    # We want shape (n_samples, N)
    # Y[k, i] ~ N(mu[i], sigma[i]^2)
    
    # Use standard normal and scale
    # This is faster than calling np.random.normal in a loop
    z = np.random.standard_normal((n_samples, n_candidates))
    Y = mu[None, :] + sigma_safe[None, :] * z
    
    # For each sample, find the index of the minimum value
    # This identifies the "best" candidate in that particular realization of the posterior
    argmins = np.argmin(Y, axis=1)
    
    # Accumulate counts
    np.add.at(counts, argmins, 1.0)
    
    # The score is the empirical probability of being the best, weighted by cost efficiency
    # We want higher score for higher probability and lower cost
    score = counts / cost_safe
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If all scores are zero (unlikely with 20 samples and N=2000, but possible if N > 20*something? No, sum of counts is 20)
    # If a candidate is never the argmin, its count is 0.
    # This is a valid exploration strategy: it ignores points that are never the best in any sample.
    # However, if the GP is very confident, the same point might be argmin every time.
    # This is fine.
    
    return score

# EVOLVE-BLOCK-END
