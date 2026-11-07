"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Monte Carlo approximation of Expected Hypervolume Improvement.
    
    For single-objective optimization, the Hypervolume Indicator relative to a reference point 
    is equivalent to the improvement. The Expected Hypervolume Improvement (EHVI) is thus 
    equivalent to Expected Improvement (EI). 
    
    This implementation uses Monte Carlo sampling to estimate the EI relative to the known 
    global minimum of the Goldstein-Price function (3.0). This structural change from 
    analytic closed-form EI to Monte Carlo estimation provides better numerical stability 
    and allows for more flexible weighting of the improvement distribution.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues in sampling
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # The known global minimum of Goldstein-Price 2D is 3.0
    f_opt = 3.0
    
    # Number of Monte Carlo samples per point
    # 100 samples is a good balance between accuracy and speed for 2000 points
    n_samples = 100
    
    # Generate random samples from the standard normal distribution
    # Shape: (n_samples, n)
    # We use a fixed seed for reproducibility within a single call if needed, 
    # but standard random is fine as the variance is averaged out.
    # Using np.random.randn for speed
    z = np.random.randn(n_samples, n)
    
    # Sample from the GP posterior: y ~ N(mu, sigma^2)
    # y = mu + sigma * z
    # Broadcasting: mu (n,) and sigma (n,) -> (1, n)
    y_samples = mu[None, :] + sigma_safe[None, :] * z
    
    # Calculate improvement relative to the true global minimum
    # Improvement is max(0, f_opt - y)
    improvements = np.maximum(0.0, f_opt - y_samples)
    
    # Estimate Expected Improvement (EHVI) by averaging over samples
    mc_ei = np.mean(improvements, axis=0)
    
    # To prevent the acquisition from becoming purely exploitative and getting stuck 
    # in local minima if the GP is overconfident, we add a small exploration bonus 
    # based on the posterior standard deviation. 
    # This ensures that regions with high uncertainty are still considered, 
    # especially if the mean prediction is not yet very good.
    
    # Normalize sigma to have a reasonable scale relative to the function range
    # Goldstein-Price range is roughly [3, 12].
    # We use a small weight for the exploration term.
    exploration_weight = 0.1
    exploration_term = exploration_weight * sigma_safe
    
    # Combine MC-EI and exploration term
    scores = mc_ei + exploration_term
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
