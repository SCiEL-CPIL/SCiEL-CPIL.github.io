"""
Acquisition function for Bayesian optimization of Weierstrass 1D.
Uses Deterministic Thompson Sampling to select the next point.
This is a structural change from deterministic bounds (UCB/EI) to
stochastic sampling, which can better handle multimodal landscapes.
"""
import numpy as np
import time

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 300  # seconds

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - START_TIME > TIME_LIMIT - 5:
        # Fallback: return UCB with a conservative beta
        # This ensures we return something valid quickly if time is running out
        sigma_safe = np.maximum(sigma, 1e-10)
        return -mu + 2.5 * sigma_safe

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Deterministic Thompson Sampling
    # We generate a random vector z. To make this reproducible without relying
    # on global np.random state (which might not be seeded by the evaluator),
    # we seed a local RandomState based on the current posterior state.
    # This ensures that the same (mu, sigma) always yields the same selection,
    # but different steps (with different mu/sigma) yield different explorations.
    
    # Create a seed from the current state
    # Using a small subset to keep it fast
    seed_val = int(np.sum(mu[:5]) * 1e6 + np.sum(sigma[:5]) * 1e3 + f_best * 100)
    # Ensure seed is within valid range for RandomState
    seed_val = seed_val % (2**32)
    
    rng = np.random.RandomState(seed_val)
    z = rng.randn(len(mu))
    
    # Sample the function value
    # f_sampled = mu + sigma * z
    sampled_f = mu + sigma_safe * z
    
    # We are minimizing the objective.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we return -sampled_f.
    scores = -sampled_f
    
    # Clip to finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
