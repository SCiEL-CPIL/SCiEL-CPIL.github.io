import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid Expected Improvement (EI) + Upper Confidence Bound (UCB) strategy
    with adaptive exploration. The EI component exploits known good regions while
    the UCB component explores high-uncertainty regions. An adaptive beta parameter
    balances exploration and exploitation based on the current state of the search.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher scores indicate
        more desirable candidates for evaluation.
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB for minimization (we want to maximize the score)
    # UCB_score = -mu + beta * sigma
    # Adaptive beta: start with higher exploration, decrease as we find better solutions
    # The range of f_best gives us a sense of how much improvement has been made
    # If f_best is close to the known minimum (0), we should exploit more
    
    # Estimate progress: how close are we to the theoretical minimum (0)?
    # Use a simple heuristic based on the scale of the problem
    # The Weierstrass function on [-0.5, 0.5] has values roughly in [0, ~2]
    # If f_best is small, we're doing well and should exploit more
    
    # Adaptive beta: interpolate between high exploration (beta=3.0) and low exploitation (beta=1.0)
    # Use f_best as a proxy for progress, clamped to [0, 1] for the interpolation
    progress = np.clip(1.0 - f_best / 2.0, 0.0, 1.0)  # 0 when f_best=2, 1 when f_best=0
    beta = 3.0 - 2.0 * progress  # beta goes from 3.0 to 1.0
    
    # UCB score for minimization
    ucb = -mu + beta * sigma
    
    # Normalize both components to have zero mean and unit variance for fair combination
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-10:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = ei - ei_mean
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std > 1e-10:
        ucb_norm = (ucb - ucb_mean) / ucb_std
    else:
        ucb_norm = ucb - ucb_mean
    
    # Combine: weight EI more heavily as we progress (exploit more late in search)
    # Early: more exploration (UCB), Late: more exploitation (EI)
    w_ei = 0.5 + 0.5 * progress  # w_ei goes from 0.5 to 1.0
    w_ucb = 1.0 - w_ei + 0.5  # w_ucb goes from 1.0 to 0.5
    
    # Final score is a weighted combination
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
