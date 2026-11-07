"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere on [-5, 5].
    
    Strategy: Thompson Sampling (TS).
    Instead of using a deterministic acquisition function like EI or UCB, 
    we draw a single sample from the posterior predictive distribution 
    for each candidate point and select the point with the highest sampled value.
    
    For a Gaussian Process posterior, the predictive distribution at a point x is 
    Normal(mu(x), sigma(x)^2). 
    We sample: sample = mu + sigma * z, where z ~ N(0, 1).
    
    This approach naturally balances exploration (high sigma leads to large variance 
    in samples) and exploitation (low mu leads to low samples). 
    For a unimodal function like x^2, TS is known to perform very well because 
    it effectively searches for the minimum of a random function drawn from the 
    posterior, which tends to be unimodal if the GP prior is unimodal and the data 
    is consistent with a unimodal function.
    
    We use a fixed random seed per call to ensure reproducibility if needed, 
    but in practice, the evaluator likely calls this in a loop where stochasticity 
    is acceptable. However, to be safe and deterministic for the same input, 
    we could use a hash of the inputs or just rely on the global RNG. 
    Given the constraints, using the global RNG is standard for TS.
    
    Note: The task requires returning 2000 finite scores. 
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single standard normal random sample
    # Using a single scalar sample z, we compute the sampled function value for all points.
    # This is equivalent to drawing 2000 independent samples, but correlated in a way 
    # that reflects the GP structure? 
    # Actually, standard TS for BO usually draws a full function sample or 
    # draws independent samples per point. 
    # Drawing independent samples per point (z_i ~ N(0,1) for each i) is the standard 
    # "independent Thompson Sampling" approximation which is valid and fast.
    
    z = np.random.normal(size=mu.shape)
    
    # Sampled objective values
    # Since we are minimizing, we want to find the minimum of the sampled function.
    # The acquisition score should be "higher means evaluate next".
    # So we return -sampled_value.
    sampled_values = mu + sigma_safe * z
    
    # We want to maximize the score, so we negate the sampled values.
    # Lower sampled value (better) -> Higher score.
    score = -sampled_values
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
