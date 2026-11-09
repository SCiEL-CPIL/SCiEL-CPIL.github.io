import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function for cost-aware Bayesian optimization.
    
    Samples from the GP posterior, ranks candidates by sampled_value / cost,
    and returns a score based on rank (higher = prefer to evaluate next).
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: sample from the GP posterior
    # The sampled value represents what we believe the objective value is
    # We want to minimize, so lower sampled values are better
    # But we also need to account for cost: prefer cheaper evaluations
    
    # Draw a single sample from the posterior for all candidates
    # This is the core of Thompson Sampling
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(n)
    
    # Cost-aware ranking: we want low sampled value AND low cost
    # Use sampled_value / cost as the ranking metric
    # Lower is better for this metric
    cost_adjusted = samples / cost
    
    # Rank candidates: lower cost_adjusted value = better rank
    # Convert to scores where higher = better
    # Use the negative of cost_adjusted so that lower values get higher scores
    score = -cost_adjusted
    
    # Check deadline
    if time.time() > deadline:
        # Return the current scores if we're running out of time
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
