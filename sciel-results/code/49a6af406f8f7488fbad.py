import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of significant improvement (10% better than f_best)
    # As f_best -> 0, this naturally decays
    threshold = 0.1 * f_best
    z_sig = (f_best - threshold - mu) / sigma_safe
    P_sig = norm.cdf(z_sig)
    
    # Combined score: blend of EI and probability-weighted improvement
    # Weight improvement magnitude by probability of achieving it
    sig_score = P_sig * np.maximum(f_best - mu, 0)
    
    # Blend: use EI for exploitation, sig_score for principled exploration
    # As f_best -> 0, EI -> 0 and sig_score -> 0, so we need to handle this
    score = (EI + sig_score) / np.maximum(cost, eps)
    
    score = np.where(np.isfinite(score), score, 0.0)
    return score
