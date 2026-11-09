import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    if _time_check():
        # Fallback to exploitation if time is running out
        return -mu

    n = len(mu)
    # Ensure sigma is positive to avoid log or division issues if used, though TS just uses it for std
    sigma = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: Draw a single sample from the posterior predictive distribution
    # f(x) ~ N(mu, sigma)
    # Since we want to MINIMIZE, we want the sample with the lowest value.
    # However, the contract says "higher meaning evaluate next".
    # So we should return the negative of the sample, or simply use -sample as the score.
    # Let's generate the sample: sample = mu + sigma * z
    z = np.random.randn(n)
    samples = mu + sigma * z
    
    # We want to select the candidate with the lowest predicted value (minimization).
    # The score should be higher for better candidates.
    # So score = -sample.
    scores = -samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
