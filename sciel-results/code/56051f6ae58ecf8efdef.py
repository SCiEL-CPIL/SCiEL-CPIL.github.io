"""Cost-aware acquisition using Thompson Sampling for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling for
    Griewank 2D Bayesian Optimization.
    
    This function samples a single function realization from the GP posterior
    and scores candidates by -f_sample / cost. This stochastic selection
    naturally couples exploration (via the posterior sample's variance) with
    cost efficiency.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Check for time budget
    elapsed = time.time() - start_time
    if elapsed > 250:
        # Fallback: use LCB approximation
        beta = 1.5
        lcb_score = beta * sigma_safe - mu
        score = -lcb_score / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Thompson Sampling: sample from N(mu, sigma^2)
    # Use a fixed seed based on f_best to ensure reproducibility but different samples
    # as f_best changes
    rng = np.random.default_rng(seed=int(abs(hash((f_best, len(mu)))) % (2**32)))
    
    # Sample a function realization
    sample = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Score by -f_sample / cost (higher is better for minimization)
    score = -sample / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
