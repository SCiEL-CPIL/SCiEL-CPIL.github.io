"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points (in objective units)
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
        For minimization, we want lower function values, so we return negative samples.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Draw a single random sample from the standard normal distribution
    # This is reused for all points to ensure the samples are correlated 
    # in the same way the posterior is (assuming the evaluator's GP implies 
    # the standard Thompson sampling approach where we sample a global offset 
    # or just sample locally). 
    # Actually, standard Thompson Sampling samples f(x) ~ N(mu(x), sigma(x)^2) for each x.
    # Since we don't have the full covariance matrix of the candidates, we cannot sample 
    # a single vector that preserves the exact joint distribution of the 2000 candidates.
    # However, the standard approximation in BO loops where only mu and sigma are passed 
    # is to sample independently: z_i ~ N(0, 1) for each point i.
    # Wait, if we sample independently, we lose the correlation structure between candidates.
    # But usually, in these simplified interfaces, "Thompson Sampling" implies 
    # sampling from the local Gaussian.
    
    # Let's generate 2000 standard normal samples
    # Using a fixed seed would be bad as it makes it deterministic and not truly random per step?
    # No, the evaluator calls this function 25 times per seed. 
    # If we use np.random.randn, it uses the global state, which is fine.
    
    z = np.random.randn(mu.shape)
    
    # Sample from the posterior: f(x) ~ N(mu(x), sigma(x)^2)
    samples = mu + sigma_safe * z
    
    # We are minimizing. The evaluator picks the candidate with the HIGHEST score.
    # Therefore, we should return negative samples so that lower function values 
    # correspond to higher acquisition scores.
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
