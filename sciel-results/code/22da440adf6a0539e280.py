import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Thompson Sampling variant (Stochastic UCB).
    Draws a single scalar sample Z ~ N(0,1) and returns -mu + sigma * Z.
    The RNG is seeded deterministically based on the inputs to ensure a consistent
    exploration path for a given GP state, leveraging findings on RNG stability.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # If we are running out of time, return a safe fallback (e.g., pure exploitation)
        # This should rarely trigger for O(N) operations, but is required.
        mu_safe = np.asarray(mu, dtype=np.float64).ravel()
        sigma_safe = np.asarray(sigma, dtype=np.float64).ravel()
        mu_clean = np.where(np.isfinite(mu_safe), mu_safe, 1e10)
        # Return negative mu to prioritize lowest mean (exploitation)
        return -mu_clean

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)

    # Determine a deterministic seed for the Thompson Sampling draw.
    # Using a hash of the data ensures that for the same GP posterior state,
    # the same exploration sample is drawn, leading to a consistent trajectory.
    # We use a simple sum-based hash for speed, mapped to a 32-bit int.
    # To avoid collisions or bias, we mix the sums of mu and sigma.
    data_hash = int(np.sum(mu_clean) * 1e6 + np.sum(sigma_safe) * 1e6)
    # Ensure positive seed
    data_hash = data_hash % (2**32)
    
    rng = np.random.default_rng(data_hash)
    z = rng.standard_normal()

    # Thompson Sampling score for minimization:
    # We want to minimize the sampled value.
    # Sampled value approx: mu - sigma * z (if we view it as UCB with beta=z)
    # Actually, standard TS for minimization:
    # Draw f ~ N(mu, sigma^2). Minimize f.
    # If we use a single global Z, f_i = mu_i - sigma_i * Z is a common proxy 
    # (equivalent to UCB with random beta).
    # Note: If Z is positive, we explore high sigma regions. If Z is negative, we exploit low mu regions.
    
    score = -mu_clean + sigma_safe * z
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
