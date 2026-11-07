# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration. The UCB component with a carefully
    tuned beta parameter helps escape local minima in the Michalewicz function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component: encourages exploration in high-uncertainty regions
    # Beta is tuned for the Michalewicz 2D landscape
    beta = 0.5
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    # Use robust normalization based on the data
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Blend EI and UCB
    # Weight EI more heavily since we want to exploit known good regions,
    # but keep UCB component for exploration
    w_ei = 0.7
    w_ucb = 0.3
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
