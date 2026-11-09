"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are finite and valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware UCB with adaptive beta
    # Early on, explore more; later, exploit more
    # f_best gives us a sense of how good we are doing
    # Use a decay: start with higher exploration, reduce over time
    
    # Estimate progress: lower f_best means we're doing well
    # Use a simple heuristic: beta decreases as f_best approaches 0
    # But we don't know the total budget used, so use f_best as proxy
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    # f_best is the lowest observed value (minimizing griewank, min is 0)
    # If f_best is close to 0, we should exploit more
    # If f_best is still high, explore more
    
    # Scale beta based on f_best
    # When f_best is large, beta is larger (more exploration)
    # When f_best is small, beta is smaller (more exploitation)
    
    # Use a smooth transition
    # beta = base_beta * (1 + alpha * f_best)
    # But f_best can be very small, so let's use a different approach
    
    # Let's use: beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    # When f_best is large, exp(-f_best/scale) is small, so beta is close to beta_min
    # Wait, that's backwards. Let me rethink.
    
    # Actually, when f_best is large (we haven't found a good minimum yet),
    # we want to explore more, so beta should be larger.
    # When f_best is small (we've found a good minimum), we want to exploit,
    # so beta should be smaller.
    
    # So: beta = beta_min + (beta_max - beta_min) * (f_best / (f_best + scale))
    # When f_best is large, f_best/(f_best+scale) ~ 1, so beta ~ beta_max
    # When f_best is small, f_best/(f_best+scale) ~ 0, so beta ~ beta_min
    
    beta_min = 0.3
    beta_max = 1.5
    scale = 0.5  # Griewank values are typically small
    
    beta = beta_min + (beta_max - beta_min) * (f_best / (f_best + scale))
    
    # Compute UCB score for minimization
    # We want to maximize: -mu + beta * sigma, divided by cost
    # This encourages low predicted values and high uncertainty, penalized by cost
    
    scores = (-mu + beta * sigma) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard: if we're taking too long, return immediately
    if time.time() - start_time > 250:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
