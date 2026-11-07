"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Deadline guard to ensure we finish well before the 330s limit
_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Strategy:
    - Draw a sample from the Gaussian posterior N(mu, sigma^2) at each candidate.
    - Since we are minimizing, we want the sample that predicts the lowest value.
    - To introduce stochasticity that helps escape local optima (as suggested by the 
      knowledge that Monte Carlo EI improved the score), we use a single sample per call.
    - However, to be more robust and exploit the structure of the sphere function, 
      we can bias the sampling slightly towards the center (0) where the minimum is known to be.
      But wait, the function is generic `acquisition(mu, sigma, f_best)`. It doesn't know 
      the coordinates. So we cannot bias towards 0 directly without coordinates.
      
    Pure Thompson Sampling:
    score_i = -sample_i
    where sample_i ~ N(mu_i, sigma_i^2)
    
    This is structurally different from the current PI+EI hybrid.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return negative mean (greedy exploitation) if out of time
        return -mu

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a sample from the posterior for each candidate
    # np.random.normal(loc=mu, scale=sigma_safe)
    # To ensure reproducibility within the function call context isn't required by the task,
    # but standard random state is used.
    
    # A single sample is the core of Thompson Sampling.
    # To potentially improve stability or performance, we could average multiple samples,
    # but that converges to the mean (EI-like). The "stochasticity" is the key.
    
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # For minimization, higher score means we prefer lower predicted values.
    # So score = -sample
    scores = -samples
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
