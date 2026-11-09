import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function.
    
    Score = -mu + beta * sigma
    
    This structurally differs from the previous EI+adaptive exploration bonus
    by using a fixed confidence parameter and a linear combination of mean
    and standard deviation, prioritizing regions with high uncertainty
    relative to the mean.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Fixed beta for LCB
    beta = 1.0
    
    # LCB score for minimization: we want to maximize -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
