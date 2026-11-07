import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle sigma = 0
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0)
    
    # Calculate Lower Confidence Bound (LCB) score
    # For minimization, we want to maximize -mu + beta * sigma
    # Using a quantile-based beta (10th percentile z-score is -1.2816, so beta = 1.2816)
    # We'll use a dynamic beta that starts high and decays as f_best improves.
    
    # f_best is the current minimum.
    # If f_best is large, we are far from optimum (0). Explore more.
    # If f_best is small, we are close. Exploit more.
    
    # Decay rate: exp(-f_best / scale)
    # Scale = 5.0 seems reasonable for [-5, 5] domain where f can be up to 25.
    beta = 1.5 * np.exp(-f_best / 5.0) + 0.5
    
    lcb_score = -mu + beta * sigma_safe
    
    # Blend EI and LCB score
    # We want to return a score where higher is better.
    # EI is in objective units. LCB is in objective units.
    # They are roughly on the same scale.
    
    # Weighting: 
    # Early on, EI might be dominated by the first few points.
    # LCB encourages exploration.
    # Let's try 0.7 * EI + 0.3 * LCB_score?
    # Or just sum them?
    
    # The prompt hint mentioned "Blending EI with ... UCB".
    # Let's use a sum with a weight.
    
    score = ei + 0.5 * lcb_score
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
