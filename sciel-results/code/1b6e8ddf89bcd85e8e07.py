import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB
    # Estimate exploration level from the average uncertainty
    # Higher average sigma implies fewer points / more uncertainty -> higher beta
    # Lower average sigma implies more points / less uncertainty -> lower beta
    
    # Normalize sigma to get a sense of scale
    # The sigma values are in objective units.
    # Typical sigma might be 0.1 - 0.5.
    # Let's use the median to be robust to outliers
    med_sigma = np.median(sigma)
    
    # Map med_sigma to beta
    # If med_sigma is 0.5, beta = 5.0
    # If med_sigma is 0.1, beta = 1.0
    # Linear interpolation
    # beta = a * med_sigma + b
    # Let's assume beta ranges from 1.0 to 5.0 for med_sigma 0.0 to 0.5
    beta = 1.0 + 8.0 * med_sigma
    
    # Cap beta to avoid extreme values
    beta = np.clip(beta, 1.0, 5.0)
    
    scores = -mu + beta * sigma
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
