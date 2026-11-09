import numpy as np
import time

# Track time to enforce the 300s deadline
_START_TIME = time.time()
_TIMEOUT_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Monte Carlo Expected Improvement (MC-EI).
    Approximates EI by drawing 50 posterior samples and averaging the improvements.
    
    Args:
        mu: array of shape (2000,) with mean predictions.
        sigma: array of shape (2000,) with standard deviation predictions.
        f_best: scalar, the best observed value so far.
        
    Returns:
        array of shape (2000,) with scores. Higher score means evaluate next.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT_LIMIT - 5.0:
        # Fallback to exploit-only if running out of time
        return -mu

    # Number of posterior samples as specified in the directive
    K = 50
    
    # Generate K standard normal samples
    z = np.random.randn(K)
    
    # Compute posterior samples for all candidates
    # Shape: (2000, 50)
    samples = mu[:, np.newaxis] + sigma[:, np.newaxis] * z[np.newaxis, :]
    
    # Compute improvement for each sample: max(0, f_best - sample)
    # Since we are minimizing, improvement is positive if sample < f_best
    improvements = np.maximum(0.0, f_best - samples)
    
    # Average improvements over the K samples
    scores = improvements.mean(axis=1)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
