import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Lower Confidence Bound (LCB) score
    # We want to maximize this score to find the best candidate
    # LCB for minimization is mu - beta * sigma. 
    # To maximize, we use -mu + beta * sigma.
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # Hybrid: 70% LCB, 30% EI
    # The LCB part drives exploration, EI drives exploitation
    scores = 0.7 * lcb_score + 0.3 * ei
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
