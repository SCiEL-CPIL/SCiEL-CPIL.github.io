import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) with Thompson Sampling
    exploration for multimodal landscapes like Rastrigin.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add Thompson Sampling exploration: sample from GP posterior
    # This helps escape local minima in multimodal landscapes
    # Use a fixed random seed for reproducibility within the acquisition call
    rng = np.random.RandomState(42)
    ts_sample = mu + sigma_safe * rng.randn(len(mu))
    
    # Combine EI and Thompson Sampling score
    # Weight: 0.7 EI + 0.3 Thompson Sampling
    # This balances exploitation (EI) and exploration (TS)
    combined_score = 0.7 * ei - 0.3 * ts_sample
    
    # Cost-aware score: Maximize combined score per unit cost
    score = combined_score / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
