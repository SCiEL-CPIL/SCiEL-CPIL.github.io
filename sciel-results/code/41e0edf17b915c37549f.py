import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses Thompson Sampling with multiple posterior samples combined with LCB.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (2000,).
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (2000,).
    f_best (float): The lowest observed value so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (2000,). Higher is better.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used too much time, fall back to simple LCB
    elapsed = time.time() - _start_time
    if elapsed > 280:  # Safety margin before 300s limit
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        beta = 0.2 * (1.0 + f_best / 10.0)
        scores = -mu + beta * sigma_safe
        return scores
    
    # Ensure sigma is non-negative and handle potential numerical issues
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling: sample from posterior and take minimum across samples
    # This provides natural exploration that scales with uncertainty
    n_samples = 10
    rng = np.random.default_rng()
    
    # Sample objective values from GP posterior: mu + sigma * N(0,1)
    # For each candidate, we want to find the lowest sampled value
    # across multiple samples, which acts as a soft-min
    sample_means = np.zeros((n_samples, len(mu)))
    for i in range(n_samples):
        z = rng.standard_normal(len(mu))
        sample_means[i] = mu + sigma_safe * z
    
    # Use log-sum-exp for numerical stability to compute soft-min
    # soft_min(x) = -log(sum(exp(-x_i)))
    neg_samples = -sample_means
    max_neg = np.max(neg_samples, axis=0, keepdims=True)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_samples - max_neg), axis=0))
    soft_min = -log_sum_exp
    
    # Combine Thompson Sampling soft-min with LCB for consistent exploitation pressure
    # Adaptive beta: more exploration early, less late
    beta = 0.2 * (1.0 + f_best / 10.0)
    lcb_scores = -mu + beta * sigma_safe
    
    # Blend: weight Thompson Sampling heavily for exploration, LCB for exploitation
    # Higher weight on soft_min means more exploration
    ts_weight = 0.7
    scores = ts_weight * (-soft_min) + (1 - ts_weight) * lcb_scores
    
    return scores
