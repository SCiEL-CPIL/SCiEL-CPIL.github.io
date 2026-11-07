import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Use Lower Confidence Bound (LCB) for minimization
    # Score is higher for lower expected values and higher uncertainty
    # This encourages exploration of uncertain regions
    beta = 2.5
    scores = (beta * sigma - mu) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
