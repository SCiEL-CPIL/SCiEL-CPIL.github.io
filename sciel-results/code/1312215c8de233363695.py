"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling with cost-awareness
    # Sample from GP posterior and compute expected cost-normalized objective
    # Use multiple samples to reduce Monte Carlo noise
    n_samples = 20
    rng = np.random.default_rng()
    
    # Generate samples: mu + sigma * Z
    # For each sample, compute cost-normalized value
    # We want to minimize, so lower is better
    # Score = -mean of (sampled_value / cost) across samples
    # This encourages low objective values AND low costs
    
    # Efficiently compute mean of (mu + sigma*Z) / cost
    # E[(mu + sigma*Z)/cost] = mu/cost + sigma*E[Z]/cost = mu/cost (since E[Z]=0)
    # But we want to capture exploration, so use actual samples
    
    # Actually, for Thompson Sampling, we sample once and use that
    # But with cost-awareness, we want to balance exploration and cost
    # Let's use: score = -E[(f(x) + cost_penalty) / cost]
    # where f(x) ~ N(mu, sigma^2)
    
    # Simpler approach: sample from posterior, then rank by sampled_value / cost
    # Take the negative of the mean of these ratios
    
    # Generate n_samples samples
    Z = rng.standard_normal((n_samples, len(mu)))
    samples = mu[None, :] + sigma_safe[None, :] * Z
    
    # Compute cost-normalized values for each sample
    cost_safe = np.maximum(cost, 1e-9)
    normalized = samples / cost_safe[None, :]
    
    # Score is negative of the mean (higher score = lower expected cost-normalized value)
    scores = -np.mean(normalized, axis=0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
