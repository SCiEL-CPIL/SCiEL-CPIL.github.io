"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling heuristic for cost-aware minimization
    # Sample from the posterior: y = mu - sigma * z
    # Since we want to minimize, we want low values.
    # The score should be "higher = evaluate next".
    # So we want to maximize (something that indicates a good candidate).
    # A good candidate for minimization has a low sampled value.
    # So we can use score = -sampled_value / cost.
    
    # Deadline guard
    start_time = time.time()
    
    # Generate random normal samples
    # To be safe with the 300s limit, we assume this vectorized operation is fast.
    # 2000 points is small.
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled function value
    # Note: sigma is standard deviation.
    # If sigma is 0, sampled value is mu.
    sigma_safe = np.maximum(sigma, 1e-12)
    sampled_y = mu - sigma_safe * z
    
    # We want to minimize sampled_y.
    # Cost is in denominator.
    # Score = -sampled_y / cost
    # If sampled_y is very negative (good), -sampled_y is large positive.
    # If cost is high, score is lower.
    
    # Ensure cost is safe
    cost_safe = np.maximum(cost, 1e-12)
    
    score = -sampled_y / cost_safe
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Return something safe if we ran out of time, though unlikely for this op
        return np.zeros_like(mu)
        
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
