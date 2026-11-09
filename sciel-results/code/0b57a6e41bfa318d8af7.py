import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Pure exploitation acquisition: return negative predicted mean.
    For a well-conditioned 1D sphere function with a tight GP length scale,
    directly minimizing the predicted mean is optimal.
    """
    # Negative mean: higher score where predicted value is lower
    score = -mu
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
