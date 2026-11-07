import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Guard against numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # z-score for improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) - standard formula for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # Smooth exploration term: sigma weighted by POI
    # This encourages exploration where we are uncertain AND likely to improve
    # (unlike UCB which explores everywhere with high sigma)
    exploration = sigma * poi
    
    # Combine EI with exploration term
    # EI is in objective units, exploration is also in objective units (sigma is std dev in objective units)
    # Weight of 0.3 balances exploitation and exploration
    score = ei + 0.3 * exploration
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
