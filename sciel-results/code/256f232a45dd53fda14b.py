import numpy as np

def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration weight scaled by f_best
    # High f_best (early) -> High beta (explore)
    # Low f_best (late) -> Low beta (exploit)
    beta = f_best
    
    # Robust quantile-normalized standard deviation
    # Use 90th percentile to better represent the scale of high-uncertainty regions
    # and avoid inflation caused by a low median in later stages
    sigma_scale = np.percentile(sigma, 90)
    if sigma_scale < 1e-9:
        sigma_scale = 1e-9
        
    sigma_norm = sigma / sigma_scale
    
    # Clip to prevent extreme outliers from dominating the score
    sigma_norm = np.clip(sigma_norm, 0, 10)
    
    # Thompson Sampling heuristic: -mu + beta * sigma_norm
    # This encourages exploring high variance regions early
    score = -mu + beta * sigma_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
