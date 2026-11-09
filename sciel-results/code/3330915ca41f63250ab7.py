"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Use a fixed RNG for reproducibility within a single acquisition call
# but we need to be careful: the evaluator calls acquisition multiple times.
# We should not use a global RNG that persists state in a way that breaks
# determinism if required, but Thompson Sampling is inherently stochastic.
# However, the task says "draws 2000 fresh uniform candidates... calls acquisition".
# It does not explicitly require determinism across calls, but standard BO loops
# might. Let's use a local RNG seeded based on the input to be deterministic
# per unique (mu, sigma, f_best) state? No, that's overkill.
# Actually, for Thompson Sampling, we just draw samples.
# To ensure we don't have issues with global state, we can create a local RNG.
# But creating a new RNG every call is cheap enough.
# However, if we want reproducibility for debugging, we might seed it.
# Let's just use np.random.default_rng() without a seed to allow exploration,
# or better, since the evaluator runs multiple seeds, the stochasticity helps.

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Thompson Sampling:
    # Draw a sample from the posterior predictive distribution for each candidate.
    # f(x) ~ N(mu(x), sigma(x))
    # We want to minimize f, so we pick the candidate with the lowest sampled value.
    # The score returned should be "higher meaning evaluate next".
    # So we return -sampled_value.
    
    # Generate random normal samples
    # Use a local random generator to avoid global state issues, but for speed,
    # np.random.standard_normal is faster.
    # Let's use np.random.standard_normal for performance.
    
    # Shape of mu and sigma is (2000,)
    # We draw 2000 samples
    samples = np.random.standard_normal(size=mu.shape)
    
    # Compute the sampled function values
    sampled_values = mu + sigma * samples
    
    # We want to minimize, so the best candidate has the lowest sampled value.
    # The acquisition score should be higher for better candidates.
    # So we negate the sampled values.
    scores = -sampled_values
    
    # Ensure finite scores
    # If sigma is 0, sampled_values = mu, which is finite.
    # If mu or sigma contain inf/nan, we should handle it.
    # The GP posterior should be well-behaved, but let's be safe.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
