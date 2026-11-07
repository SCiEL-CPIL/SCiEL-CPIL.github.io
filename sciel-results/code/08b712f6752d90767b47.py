"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for safety
_start_time = time.time()
_DEADLINE = 330.0  # Seconds

# Thompson Sampling uses a random seed for the noise realization.
# We use a fixed seed to ensure reproducibility in the "random" draws of the TS noise,
# although the GP posterior itself changes with data, the TS noise draw is a single scalar 
# (or vector) per step. 
# Actually, TS requires sampling a value from N(mu, sigma^2) for each candidate.
# To do this efficiently and consistently, we can generate a standard normal vector 
# once per acquisition call.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Thompson Sampling.
    
    For minimization of f, we want to find low f.
    The GP predicts mean mu and std sigma for the objective f.
    
    Thompson Sampling strategy:
    1. Sample a single function realization from the posterior: 
       f_sample(x) ~ N(mu(x), sigma(x)^2)
    2. Select the candidate x with the lowest f_sample(x).
    
    This is equivalent to maximizing the score:
    score(x) = -f_sample(x) = -mu(x) - sigma(x) * z
    
    where z is a standard normal random variable.
    Note: Since we sample f_sample(x) = mu(x) + sigma(x) * z,
    the point with the minimum f_sample is the one we want.
    Therefore, to return a score where "higher is better", we return -f_sample.
    
    Parameters:
    mu: array of shape (2000,) - GP posterior mean
    sigma: array of shape (2000,) - GP posterior std dev
    f_best: float - best observed value so far (lowest)
    
    Returns:
    scores: array of shape (2000,) - higher is better
    """
    # Check time budget
    current_time = time.time()
    if current_time > _DEADLINE - 5.0:
        # Fallback to a simple exploitation strategy if running out of time
        # to avoid timeout. We want low mu, so high score = -mu.
        mu_arr = np.asarray(mu, dtype=np.float64).ravel()
        return -mu_arr

    # Ensure inputs are numpy arrays and 1D
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Safety: sigma should be non-negative. Clip small negative values due to numerical noise.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate a standard normal random variable for the Thompson Sampling noise.
    # We use a single scalar z for the entire batch to maintain the correlation structure
    # implied by the GP posterior (all points are sampled from the same function realization).
    # Note: In a full TS implementation with a GP, one would sample a function value 
    # from the joint distribution. However, since we are given marginal mu and sigma 
    # at 2000 points, and the GP is Gaussian, the joint distribution of these 2000 points 
    # is multivariate normal. 
    # Sampling a single realization f(x_i) for all i requires the covariance matrix.
    # We do NOT have the covariance matrix, only marginals.
    # Therefore, we must approximate the sample.
    # A common approximation when only marginals are available is to use:
    # f_sample(i) = mu(i) + sigma(i) * z
    # This assumes the noise is identical across all points (perfect correlation).
    # This is a valid "Thompson Sampling" variant often used when full covariance is unavailable,
    # effectively sampling a function from the set of functions consistent with the marginals
    # under a "global shift" noise model. 
    # Alternatively, independent noise: f_sample(i) = mu(i) + sigma(i) * z_i
    # Independent noise is generally preferred for exploration diversity if the correlation
    # structure is not known or is weak. Given the length scale is 0.2 in 6D, 
    # correlations between random uniform points are likely weak.
    # Let's use independent standard normal vectors z for each point.
    
    # Use a fixed random state for reproducibility of the "noise draw" given the inputs?
    # No, TS should be stochastic. But the evaluator runs multiple seeds.
    # The randomness comes from the initial points and the TS draw.
    # We should use a fresh random draw each time to allow exploration.
    
    z = np.random.randn(len(mu))
    
    # Sampled function values
    f_samples = mu + sigma_safe * z
    
    # We want to minimize f. So the "best" point has the lowest f_sample.
    # The score to maximize is -f_sample.
    scores = -f_samples
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
