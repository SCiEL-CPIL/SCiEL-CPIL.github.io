"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: Sample from the posterior distribution
    # N(mu, sigma^2) at each candidate location.
    # The candidate with the highest sample value is selected.
    # This is equivalent to maximizing the expected value under the uncertainty.
    # For minimization, we sample f ~ N(mu, sigma) and want to find the lowest f.
    # However, the task states "higher meaning evaluate next".
    # So we should return scores such that the "best" (lowest objective) candidate has the highest score.
    # Let's sample s ~ N(mu, sigma).
    # If we return s, we would pick the highest s, which is bad for minimization.
    # We should return -s? No, let's think about the standard TS for minimization.
    # Usually, for minimization, we sample f ~ N(mu, sigma) and pick argmin f.
    # The acquisition function returns scores where higher is better.
    # So we want the candidate with the lowest sampled f to have the highest score.
    # Therefore, we should return -sampled_f.
    
    # Generate random normal variables
    # Use a fixed seed? No, the evaluator runs the loop, and we want stochasticity to escape local optima.
    # However, if we use random numbers, the result is non-deterministic.
    # The task says "For each seed 0, 1 and 2...". This implies the evaluator might control randomness or expect determinism?
    # "The evaluator starts from 5 uniform random points... draws 2000 fresh uniform candidates...".
    # The acquisition function is called with mu, sigma, f_best.
    # If I use np.random.randn, it consumes the global RNG state.
    # Is that allowed? Yes, standard library and NumPy are available.
    # But non-determinism might be risky if the evaluator expects consistent behavior for a given input?
    # "mu and sigma have shape (2000,). Return 2000 finite scores".
    # If I return random scores, the "best" candidate changes every time even if mu/sigma are the same.
    # This is the nature of Thompson Sampling.
    
    # Let's check the constraint: "higher meaning evaluate next".
    # We want to minimize f.
    # Sample f_sampled = mu + sigma * z, where z ~ N(0,1).
    # We want to pick the candidate with the smallest f_sampled.
    # So we return scores = -f_sampled.
    
    z = np.random.randn(mu.shape[0])
    f_sampled = mu + sigma * z
    return -f_sampled
# EVOLVE-BLOCK-END
