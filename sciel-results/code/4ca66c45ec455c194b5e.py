"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for posterior sampling to avoid redundant work within a single acquisition call
# The acquisition function is called once per step with 2000 candidates.
# We compute the posterior samples once per call.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using posterior sampling approximation to Expected Improvement.
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (n_candidates,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far (minimum since we're minimizing).
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means more promising.
    """
    n_candidates = len(mu)
    
    # Handle edge cases
    if n_candidates == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Number of posterior samples
    n_samples = 50
    
    # Draw samples from the Gaussian process posterior
    # For each candidate i and sample j: f_sample[i,j] ~ N(mu[i], sigma[i]^2)
    # We can vectorize this: 
    #   samples = mu[:, None] + sigma[:, None] * randn(n_candidates, n_samples)
    
    # Use a fixed random state for reproducibility within a call? 
    # Actually, the task says "draws 2000 fresh uniform candidates" each step,
    # but the posterior is given. The randomness in posterior sampling is our choice.
    # We should use random samples to get a stochastic acquisition.
    
    rng = np.random.default_rng()
    
    # Generate random normal samples
    # Shape: (n_candidates, n_samples)
    z = rng.standard_normal((n_candidates, n_samples))
    
    # Posterior samples
    samples = mu[:, None] + sigma[:, None] * z
    
    # For each candidate, compute the expected improvement over f_best
    # EI = E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    # Using samples: EI_approx = mean(max(f_best - samples, 0), axis=1)
    
    improvement = np.maximum(f_best - samples, 0.0)
    scores = np.mean(improvement, axis=1)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
