import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Thompson Sampling variant with exploitation bias.
    
    Uses stochastic posterior sampling to escape local minima in the multimodal
    Michalewicz landscape, combined with a small deterministic exploitation bias
    toward the known best value to maintain convergence pressure.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior at candidate points
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed objective value so far
    
    Returns
    -------
    np.ndarray, shape (N,)
        Acquisition scores; higher scores are preferred.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample from posterior predictive distribution
    # For minimization, we want low values. Score = -(mu - sigma * z) = -mu + sigma * z
    # where z ~ N(0,1). Higher score means more likely to be a good (low) value.
    z = np.random.standard_normal(mu.shape)
    ts_score = -mu + sigma_safe * z
    
    # Compute Expected Improvement as a deterministic exploitation component
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Blend Thompson sampling with EI exploitation
    # Weight primarily on TS for exploration, with a smaller EI component
    # for convergence pressure toward the known best
    w_ts = 0.8
    w_ei = 0.2
    
    score = w_ts * ts_score + w_ei * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If we're running out of time, return immediately
    if time.time() > deadline:
        return score
    
    return score
