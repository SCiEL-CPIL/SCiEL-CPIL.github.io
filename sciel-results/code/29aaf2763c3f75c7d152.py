# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses Thompson Sampling: draw a single scalar sample z ~ N(0,1) and
    compute scores = -(mu - sigma * z). Higher scores indicate candidates
    more likely to have low objective values, balancing exploration and
    exploitation without a manually tuned beta parameter.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw one sample from standard normal
    z = np.random.randn()
    
    # Posterior predictive sample: mu - sigma * z
    # We want to minimize, so higher score = more promising
    # score = -(mu - sigma * z) = -mu + sigma * z
    scores = -mu + sigma * z
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
