"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Multi-sample Thompson Sampling acquisition function.
    
    mu: array of shape (2000,) - posterior mean at candidates
    sigma: array of shape (2000,) - posterior std at candidates
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    n_candidates = len(mu)
    n_samples = 100
    
    # Generate random samples from the marginal posterior at each candidate
    # Shape: (n_samples, n_candidates)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, n_candidates))
    
    # Thompson Sampling: 
    # In each sample, the "best" candidate is the one with the minimum value.
    # We estimate the probability of each candidate being selected by counting
    # how often it is the minimum across the n_samples draws.
    # This provides a score where higher values indicate a higher probability
    # of being the next point to evaluate, which aligns with the TS policy.
    
    # Find the index of the minimum value for each sample
    # samples has shape (n_samples, n_candidates)
    # argmin along axis=1 gives the index of the min for each row (sample)
    min_indices = np.argmin(samples, axis=1)
    
    # Count the occurrences of each candidate index in min_indices
    # This gives the number of times each candidate was the minimum
    scores = np.bincount(min_indices, minlength=n_candidates).astype(np.float64)
    
    return scores
# EVOLVE-BLOCK-END
