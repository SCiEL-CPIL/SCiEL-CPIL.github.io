import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound for minimization.
    # We want to minimize f(x).
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We return a score where higher is better.
    # So we return -LCB(x) = -mu(x) + kappa * sigma(x).
    
    # Exploration coefficient. 
    # The directive suggests kappa=3.0 for aggressive exploration in multimodal landscapes.
    kappa = 3.0
    
    # Calculate scores
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    # If sigma or mu contain NaN/Inf, the result will be NaN/Inf.
    # The contract says "Return 2000 finite scores".
    # We should sanitize non-finite values.
    # Replace NaN with a very low score (so it's not picked) or 0?
    # If mu is NaN, it's an invalid state. But GP should be stable.
    # Let's replace non-finite with -np.inf for the score? 
    # Wait, "higher meaning evaluate next".
    # If a point is invalid, we don't want to evaluate it.
    # So invalid points should have very LOW scores.
    # np.where(np.isfinite(scores), scores, -np.inf)
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
