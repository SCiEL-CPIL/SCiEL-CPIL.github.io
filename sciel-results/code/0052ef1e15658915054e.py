import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s buffer
    
    # Avoid division by zero
    cost_safe = cost + 1e-9
    sigma_safe = sigma + 1e-9
    
    # Thompson Sampling:
    # Sample a single realization from the posterior GP.
    # Since we only have mu and sigma (marginal posteriors), we approximate
    # the joint sample by independent Gaussian samples for each candidate.
    # This is a standard approximation when covariance information is not passed,
    # or when we treat the provided sigma as the full uncertainty for selection.
    
    # To be more robust and potentially capture some correlation if we could,
    # but here we stick to independent samples as per the available data.
    # We sample f(x) ~ N(mu, sigma^2)
    
    # Use a fixed random seed? No, Thompson sampling benefits from randomness
    # across calls if the function were stateful, but here it's stateless.
    # However, since the evaluator calls this function repeatedly in a loop
    # (implied by "loop that minimizes"), and the function is stateless,
    # using a global random state or np.random default is fine.
    # But wait, if the function is called 2000 times? No, the contract says:
    # "evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    # So acquisition is called ONCE per iteration with arrays of size 2000.
    # Therefore, we generate 2000 samples at once.
    
    # Generate samples
    # We want to minimize the cost-adjusted value.
    # Cost-aware Thompson Sampling: minimize sample_value / cost
    
    # To avoid issues with very small costs or values, we can simply compute
    # the ratio.
    
    # Draw samples
    # np.random.normal(loc=mu, scale=sigma_safe)
    
    # We need to be careful with performance.
    # 2000 samples is very fast.
    
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Cost-adjusted score: we want to maximize the "desirability".
    # Desirability is high when (sample / cost) is low.
    # So we can return -samples / cost_safe.
    # Or, to make it more "acquisition-like" (higher is better),
    # we can just return -samples / cost_safe.
    
    # However, sometimes pure TS can be too aggressive.
    # The prompt suggests: "divide this sampled objective value by the evaluation cost"
    # and "selecting the candidate that minimizes this sample".
    
    # So, minimize (sample / cost).
    # Acquisition function returns scores where higher = evaluate next.
    # So score = - (sample / cost)
    
    scores = -samples / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check time (though this is very fast, it's good practice)
    if time.time() > deadline:
        # Fallback to a simple exploitation-based score if somehow we timed out
        # (highly unlikely for 2000 ops)
        return -mu / cost_safe
    
    return scores
