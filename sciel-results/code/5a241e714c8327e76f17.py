"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to enforce deadline
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Instead of maximizing Expected Improvement (deterministic), this method
    samples from the posterior predictive distribution and selects the point
    with the lowest sampled value. This is a structural change from the
    deterministic EI approach, offering a different exploration-exploitation
    balance that can escape local optima more effectively in some scenarios.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: return random scores or zeros if we are about to be killed
        # Returning zeros means the evaluator will pick index 0 (or arbitrary)
        # This is a safety net, unlikely to trigger.
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive to avoid division by zero or invalid samples
    sigma = np.maximum(sigma, 1e-6)
    
    # Generate samples from the Gaussian posterior
    # mu and sigma have shape (2000,)
    samples = mu + sigma * np.random.randn(mu.shape[0])
    
    # The evaluator picks the candidate with the HIGHEST score.
    # We want to evaluate the candidate with the LOWEST sampled value (minimization).
    # Therefore, we return the negative of the samples.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
