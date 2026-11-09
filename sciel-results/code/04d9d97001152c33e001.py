# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient acquisition function using Monte Carlo estimation.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best observed value so far
        
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher means more promising
    """
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Select top-K candidates for detailed KG estimation
    # Using a UCB-like score for selection to balance exploitation and exploration
    k = 100
    selection_score = mu - 1.0 * sigma_safe
    top_k_indices = np.argsort(selection_score)[:k]
    
    # Initialize scores with EI for all candidates
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    scores = ei.copy()
    
    # 2. Monte Carlo estimation for top-K candidates
    # Independent KG approximation
    n_mc = 200
    rng = np.random.default_rng(42)
    
    mu_top = mu[top_k_indices]
    sigma_top = sigma_safe[top_k_indices]
    
    for i in range(k):
        # Sample f_i
        f_i_samples = rng.normal(loc=mu_top[i], scale=sigma_top[i], size=n_mc)
        
        # Sample others
        # Create mask for others
        others_idx = np.concatenate((np.arange(0, i), np.arange(i+1, k)))
        mu_others = mu_top[others_idx]
        sigma_others = sigma_top[others_idx]
        
        others_samples = rng.normal(loc=mu_others, scale=sigma_others, size=(n_mc, len(others_idx)))
        min_others = np.min(others_samples, axis=1)
        
        # Calculate gain
        new_min = np.minimum(f_i_samples, min_others)
        gain = np.maximum(0, f_best - new_min)
        
        scores[top_k_indices[i]] = np.mean(gain)
        
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
