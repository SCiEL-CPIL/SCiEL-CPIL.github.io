import numpy as np
import time

_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling cost-aware acquisition for Bayesian optimization (Levy 2D).
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition values (higher = better)
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    # Deadline guard: if we're close to the 330s kill limit, return quickly
    if elapsed > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Simple Thompson sample
        rng = np.random.default_rng(42)
        z = rng.standard_normal(mu.shape)
        sample = mu - sigma_safe * np.abs(z)
        improvement = np.maximum(f_best - sample, 0.0)
        score = improvement / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: draw a posterior sample for each candidate
    # For minimization, the posterior sample is: mu - sigma * |z| where z ~ N(0,1)
    # This gives a random draw from the predictive distribution
    # Use a fixed seed for reproducibility within a single call, but vary across calls
    # Actually, we want randomness across iterations. Use time-based seed.
    # But we need finite, deterministic output for the 2000 candidates in one call.
    # The randomness comes from the GP fit (mu, sigma change each iteration).
    # For Thompson Sampling, we sample once per candidate per acquisition call.
    
    # Generate standard normal samples
    rng = np.random.default_rng(int(elapsed * 1000) % (2**31))
    z = rng.standard_normal(mu.shape)
    
    # Posterior sample: for minimization, lower is better
    # Sample from N(mu, sigma^2): sample = mu + sigma * z
    # But we want to score by expected improvement, so use the sample directly
    posterior_sample = mu + sigma_safe * z
    
    # Improvement over f_best (for minimization, lower f is better)
    # If sample < f_best, improvement = f_best - sample
    # Otherwise improvement = 0
    improvement = np.maximum(f_best - posterior_sample, 0.0)
    
    # Cost-aware score: improvement per unit cost
    # This directly optimizes expected simple regret per evaluation cost
    score = improvement / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
