import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) score
    # Maximizing (beta * sigma - mu) is equivalent to minimizing (mu - beta * sigma)
    beta = 2.5
    lcb_component = beta * sigma_safe - mu
    
    # Combine EI and LCB
    # The LCB term adds an exploration bonus proportional to uncertainty
    # and a penalty for high mean, reinforcing the EI's preference for low mean
    # but with a different weighting of uncertainty.
    score = ei + 0.5 * lcb_component
    
    return score
