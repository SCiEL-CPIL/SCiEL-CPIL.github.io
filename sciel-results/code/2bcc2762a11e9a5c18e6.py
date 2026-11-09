import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Monte Carlo Thompson Sampling
    with an exponential cost gate.
    
    Score = (f_best - min_sampled_value) * exp(-k * cost)
    
    This uses a stochastic approach (Thompson Sampling) rather than the
    deterministic Expected Improvement. It explores regions with high
    uncertainty by sampling from the posterior.
    
    Args:
        mu: array of GP means, shape (N,)
        sigma: array of GP standard deviations, shape (N,)
        f_best: scalar, best function value observed so far
        cost: array of costs, shape (N,)
        
    Returns:
        scores: array of acquisition values, shape (N,), higher is better
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margin check
    if time.time() - start_time > 290:
        return np.zeros_like(mu)

    # Prevent division by zero / invalid std
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Number of Monte Carlo samples for Thompson Sampling
    # 10 samples is a good balance between accuracy and speed for N=2000
    n_samples = 10
    
    # Draw samples from the GP posterior
    # Shape: (n_samples, N)
    samples = np.random.normal(mu, sigma_safe, size=(n_samples, len(mu)))
    
    # For minimization, we are interested in the lowest sampled value per point
    # min_samples[i] is the best (lowest) value observed for point i across samples
    min_samples = np.min(samples, axis=0)
    
    # Calculate the improvement over f_best
    # If min_samples < f_best, improvement is positive
    improvement = f_best - min_samples
    
    # Clip negative improvements to 0 (no improvement)
    improvement = np.maximum(improvement, 0.0)
    
    # Structural Change: Multiplicative Exponential Cost Gate
    # k calibrated for cost range [1, 2]
    k = 1.5
    cost_gate = np.exp(-k * cost)
    
    scores = improvement * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final time check
    if time.time() - start_time > 295:
        pass
         
    return scores
