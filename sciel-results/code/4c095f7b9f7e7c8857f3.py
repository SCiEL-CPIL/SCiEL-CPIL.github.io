import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a hybrid of Expected Improvement (EI) 
    and a quantile-based probability of improvement with margin.
    
    The quantile-based score is P(f(x) < f_best - epsilon) where epsilon is 
    a fraction of |f_best|, creating a robust bounded score.
    
    We blend EI and the probability score to balance exploitation and exploration.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate the Z-score for standard EI
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Quantile-based probability of improvement with margin
    # epsilon is a margin: we want P(f(x) < f_best - epsilon)
    # For Ackley, f_best can be close to 0, so use a relative margin
    # If f_best is very small, use a small absolute margin
    epsilon = 0.1 * max(abs(f_best), 0.1)
    
    # Z-score for the margin-based probability
    z_margin = (f_best - epsilon - mu) / sigma_safe
    
    # Probability of improvement by at least epsilon
    prob_improve = norm.cdf(z_margin)
    prob_improve = np.clip(prob_improve, 0.0, 1.0)
    
    # Blend EI and probability score
    # Normalize EI to [0, 1] range for blending
    # Use a scaling factor based on the maximum EI
    max_ei = np.max(ei)
    if max_ei > 1e-12:
        ei_norm = ei / max_ei
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend: 0.7 * EI + 0.3 * probability
    # This gives more weight to EI (exploitation) while incorporating the robust probability
    score = 0.7 * ei_norm + 0.3 * prob_improve
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return score
