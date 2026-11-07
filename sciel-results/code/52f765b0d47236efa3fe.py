"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of Shekel 4d using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard: track elapsed time to avoid timeout
    t0 = time.time()
    
    # Ensure finite inputs and proper shapes
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Handle edge cases: sigma can be very small or zero
    # Add a small epsilon to avoid division by zero or log(0) issues if we were doing logs, 
    # but here we just need it for sampling.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-awareness:
    # The cost is 1 + mean(x_scaled). Lower cost is better.
    # We want to prioritize low cost. 
    # A common way is to divide the acquisition value by the cost, 
    # or subtract a penalty. 
    # Dividing EI/TS by cost is a standard heuristic for cost-aware BO.
    # However, Thompson Sampling returns a sample from the objective function.
    # If we divide the sample by cost, we are effectively optimizing sample/cost.
    # This encourages finding good values with low cost.
    
    cost_safe = np.maximum(cost, 1.0) # Cost is at least 1.0 per problem statement
    
    # Thompson Sampling:
    # Draw a sample from the predictive distribution N(mu, sigma^2)
    # To make the selection diverse and avoid getting stuck, we use a single sample.
    # However, the acquisition function is called once per iteration to score 2000 candidates.
    # The evaluator picks the candidate with the highest score.
    # If we return a single sample, the "best" candidate is the one with the highest sampled value.
    # This is equivalent to Thompson Sampling.
    
    # Generate random noise
    noise = np.random.randn(len(mu))
    
    # Sample from the posterior predictive distribution
    # Since we are minimizing, a lower value is better.
    # But the acquisition score must be HIGHER = better.
    # So we should return -sample (negative of the predicted objective).
    # Or, we can think of it as maximizing -f(x).
    # Let y ~ N(mu, sigma^2). We want to minimize y.
    # So we want to maximize -y.
    # Score = -y / cost? Or just -y?
    
    # If we use -y / cost, we are maximizing the expected improvement per unit cost in a stochastic sense.
    # Let's try Score = -sample / cost_safe.
    
    sample = mu + sigma_safe * noise
    scores = -sample / cost_safe
    
    # Ensure all scores are finite
    # If sigma is huge, sample might be huge, but cost is bounded.
    # If mu is huge, sample is huge.
    # The problem guarantees finite inputs usually, but let's be safe.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check time budget
    elapsed = time.time() - t0
    # The computation is O(N) and very fast for N=2000. 
    # No complex loops needed.
    
    return scores
# EVOLVE-BLOCK-END
