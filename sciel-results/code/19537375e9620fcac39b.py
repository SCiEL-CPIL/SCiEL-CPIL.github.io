"""
Cost-aware acquisition for Bayesian optimization (hartmann 3d).
Strategy: Cost-Aware Thompson Sampling.
Instead of deterministic EI/cost, we sample from the posterior to estimate
the expected improvement per unit cost. This introduces stochasticity that
helps escape local optima in the acquisition landscape, which is often
sharper than the objective landscape.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, best observed value so far
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    score: (2000,) array of acquisition scores (higher is better)
    """
    t0 = time.time()
    deadline = t0 + 295.0  # Safety margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate random samples from the posterior: y ~ N(mu, sigma^2)
    # This is the core of Thompson Sampling
    rng = np.random.default_rng()
    samples = rng.normal(mu, sigma_safe)
    
    # The "value" of a point in Thompson Sampling is the negative of the sampled function value
    # We want to minimize f, so we maximize -f.
    # However, we need to account for cost.
    
    # Cost-aware Thompson Sampling:
    # We can view this as maximizing the expected improvement per unit cost.
    # Alternatively, we can use a "cost-normalized" sample.
    # Let's use the heuristic: Score = (f_best - sample) / cost
    # This approximates the Expected Improvement per unit cost for the specific sample drawn.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate the "improvement" of the sample over the current best
    improvement = f_best - samples
    
    # Only positive improvements contribute to the score (or we can keep it raw, 
    # but EI usually clamps at 0). In TS, we usually just pick the sample with min value.
    # But with cost, we want min value per cost.
    # Score = improvement / cost.
    # If improvement is negative (sample is worse than best), score is negative.
    # This is fine, as we pick the max score.
    
    score = improvement / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard: if we're close to deadline, return immediately
    if time.time() > deadline:
        return score
        
    return score
# EVOLVE-BLOCK-END
