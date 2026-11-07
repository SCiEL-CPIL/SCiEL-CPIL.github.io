import time
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Linear Lower Confidence Bound (LCB) for minimization.
    
    LCB is a linear combination of the mean and standard deviation.
    For minimization, we aim to minimize $\mu(x) - \beta \sigma(x)$.
    To fit the interface where higher scores are preferred, we return:
    score(x) = -(\mu(x) - \beta \sigma(x)) = -\mu(x) + \beta \sigma(x).
    
    This encourages exploration in high-uncertainty regions (high sigma) 
    and exploitation in low-prediction regions (low mu), which is effective 
    for multimodal functions like Weierstrass.
    """
    start_time = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Exploration-exploitation trade-off parameter
    # A higher beta favors exploration. 
    # Weierstrass is multimodal, so we use a relatively high beta.
    beta = 2.0
    
    # Calculate LCB scores
    # We want to maximize the score, so we negate the LCB objective (mu - beta*sigma)
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    # Time guard
    if time.time() - start_time > 290:
        return scores

    return scores
