import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and Lower Confidence Bound (LCB)
    using standardized scores to balance exploitation and exploration.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle numerical stability
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative, Phi(z) is small, EI is small.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate LCB Score (for minimization, we want low mu and high sigma)
    # LCB value = mu - beta * sigma
    # We want to MAXIMIZE the score, so we use -LCB = -mu + beta * sigma
    beta = 1.0
    lcb_score = -mu + beta * sigma_safe
    
    # 3. Standardize both scores to have mean 0 and std 1
    # This allows combining them regardless of their original scales
    def standardize(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < eps:
            return arr - mean
        return (arr - mean) / std
    
    ei_std = standardize(ei)
    lcb_std = standardize(lcb_score)
    
    # 4. Combine with weights
    # Weighting EI more heavily usually leads to better convergence,
    # but LCB helps escape local optima in multi-modal functions like ST.
    w_ei = 0.6
    w_lcb = 0.4
    
    scores = w_ei * ei_std + w_lcb * lcb_std
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
