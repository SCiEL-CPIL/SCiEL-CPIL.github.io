"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_LIMIT = 320.0  # Safety margin before 330s kill

def _check_deadline():
    return (time.time() - _start_time) > _TIMEOUT_LIMIT

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling strategy:
    # Instead of calculating a deterministic score like EI or UCB,
    # we sample a single value from the posterior distribution N(mu, sigma^2)
    # for each candidate. We then select the candidate with the lowest sampled value.
    # This is a stochastic approach that naturally balances exploration (high sigma
    # leads to high variance in samples, potentially yielding very low values)
    # and exploitation (low mu leads to lower expected samples).
    
    # To make it a valid "acquisition function" that returns scores where higher is better,
    # we can return the negative of the sampled value.
    # However, standard Thompson Sampling usually involves drawing one sample per candidate
    # and picking the argmin. 
    # The interface requires returning 2000 scores. 
    # If we just return -sample, the evaluator picks the max score (which corresponds to min sample).
    
    # We must ensure finite scores.
    # sigma is standard deviation.
    
    # Check for deadline to avoid timeout
    if _check_deadline():
        # Return a valid fallback: pure exploitation (lowest mu)
        # Higher score is better, so we want high score for low mu.
        # Return -mu.
        return -mu
    
    # Sample from N(mu, sigma^2)
    # Use a fixed random seed per call? No, the function is called once per iteration.
    # We need a single sample per candidate for this iteration.
    # np.random.normal is used.
    
    # Handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw one sample for each of the 2000 candidates
    samples = np.random.normal(mu, sigma_safe)
    
    # The acquisition score should be such that the candidate with the best (lowest)
    # objective value has the highest score.
    # So, score = -sample.
    
    # Ensure finiteness
    scores = -samples
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
