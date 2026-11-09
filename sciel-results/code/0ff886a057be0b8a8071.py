"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Use a fixed seed for Thompson Sampling to be deterministic across calls
# within a single evaluation run, but the evaluator calls acquisition fresh each step.
# We use a global RNG state that persists, or we use a seed based on call count.
# However, the task says "draws 2000 fresh uniform candidates... calls acquisition".
# If we use random samples, we need them to be consistent? No, the evaluator picks the argmax.
# If we use random samples, the result is stochastic.
# Standard Thompson Sampling uses a random draw.
# To ensure reproducibility or stability, we might want to use a fixed seed per call?
# No, if we use the same random draw every time, it's deterministic.
# If we use a new random draw each time, it's stochastic.
# The task doesn't forbid stochasticity.
# However, "higher meaning evaluate next".
# Let's use a simple Gaussian sample.

# To avoid issues with stochasticity affecting the score (variance), 
# we can use a single sample.
# But wait, if the evaluator runs the whole loop, the stochasticity might help or hurt.
# Let's stick to the standard TS: sample from posterior.

# One concern: if sigma is very small, the sample is close to mu.
# If sigma is large, the sample is spread out.

# Let's add a deadline guard as per directive.
# The function is called 25 times (30 evals - 5 initial).
# Total time budget 300s.
# We should check time at the start.

_start_time = time.time()
_TIME_LIMIT = 290.0  # Safety margin

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _TIME_LIMIT:
        # Return deterministic fallback: Expected Improvement
        # to ensure we don't crash or return garbage
        from scipy.stats import norm
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Thompson Sampling: Sample from the posterior distribution
    # y ~ N(mu, sigma^2)
    # We want to minimize, so we want the lowest y.
    # The acquisition score should be "higher is better".
    # So we return -y.
    
    # Use a fixed seed? No, let's use a global RNG to maintain state across calls if needed, 
    # or just default. Default numpy RNG is fine.
    # However, to be safe and potentially more stable, we could use a single sample.
    
    # Draw samples
    # shape: (2000,)
    samples = mu + sigma * np.random.standard_normal(len(mu))
    
    # We want to minimize the function value.
    # So the "score" for ranking is the negative of the sample.
    # Higher score = lower sample value.
    scores = -samples
    
    # Ensure finite scores
    # If sigma is inf or mu is inf, samples might be inf.
    # The problem states mu and sigma are from a GP, so they should be finite.
    # But let's be safe.
    if not np.all(np.isfinite(scores)):
        # Fallback to EI if non-finite values occur
        from scipy.stats import norm
        z = (f_best - mu) / (sigma + 1e-9)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        # Replace any non-finite in fallback too
        scores[~np.isfinite(scores)] = 0.0
        
    return scores
# EVOLVE-BLOCK-END
