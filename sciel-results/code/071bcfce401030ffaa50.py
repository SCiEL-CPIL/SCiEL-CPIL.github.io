import numpy as np
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time at the start of each acquisition call and bail out early
# if we're getting too close to the limit, returning a safe fallback.
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 330s limit

# Fixed seed for deterministic Thompson Sampling draws per acquisition call.
# This ensures reproducibility across runs while allowing the "randomness"
# to come from the fixed set of random numbers, which is standard for
# deterministic Thompson Sampling implementations in competitive settings.
_TS_SEED = 42
_rng = np.random.default_rng(_TS_SEED)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization
    of Goldstein-Price 2D.
    
    Instead of combining deterministic acquisition functions (like EI or UCB),
    this method samples a single function value from the posterior predictive
    distribution N(mu, sigma^2) at each of the 2000 candidates. The candidate
    with the minimum sampled value is selected. This inherently balances
    exploration (high sigma leads to high variance in samples, potentially
    yielding very low values) and exploitation (low mu leads to low samples).
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better. We return -sampled_value.
    """
    global _start_time, _DEADLINE_MARGIN, _rng
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple fallback: minimize mu, slightly penalize sigma? 
        # Actually, if we are close to deadline, we just want a valid finite array.
        # Using -mu is a safe exploitation-only fallback.
        mu_arr = np.asarray(mu, dtype=np.float64)
        return np.nan_to_num(-mu_arr, nan=0.0, posinf=-1e6, neginf=1e6)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero or invalid values in normal dist
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate standard normal random numbers for Thompson Sampling
    # We draw exactly 'n' samples. Using a fixed seed per call would make
    # the same candidates always get the same random offset, which might
    # bias towards specific coordinates if the candidate set is static.
    # However, the candidate set changes every step (fresh uniform candidates).
    # But the *positions* of the candidates in the array might correlate with
    # the random draw if we just use a global RNG state that advances.
    # To ensure independence from array index, we should draw fresh randoms.
    # Since we use a global _rng, the sequence depends on how many times it's been called.
    # This is generally fine for BO.
    
    z = _rng.standard_normal(n)
    
    # Sampled function values
    sampled_values = mu + sigma_safe * z
    
    # We want to minimize the function, so the "score" should be higher for lower values.
    # Thus, we return the negative of the sampled value.
    scores = -sampled_values
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
