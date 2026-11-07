import numpy as np


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # UCB for minimization: maximize -mu + beta * sigma
    # Adaptive beta: start higher for exploration, decrease as we converge
    # Use the spread of sigma to calibrate beta
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Beta adapts: larger when sigma is more variable (early exploration)
    # Smaller when sigma is uniform (later exploitation)
    # Normalize sigma variation: if sigma_std is high relative to sigma_mean, explore more
    if sigma_mean > 1e-12:
        cv = sigma_std / (sigma_mean + 1e-12)  # coefficient of variation
    else:
        cv = 0.0
    
    # Map cv to beta: cv=0 -> beta~1.5, cv=1 -> beta~4.0
    beta = 1.5 + 2.5 * np.clip(cv, 0.0, 1.0)
    
    # Also scale beta by how far f_best is from 0 (the minimum)
    # When f_best is large, we're early, explore more
    # When f_best is small, we're close, exploit more
    # f_best for Weierstrass on [-0.5,0.5] ranges from ~5+ down to 0
    f_best_scale = np.clip(f_best, 0.0, 10.0)
    beta *= (0.5 + 0.5 * f_best_scale / (1.0 + f_best_scale))
    
    score = -mu + beta * sigma
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
