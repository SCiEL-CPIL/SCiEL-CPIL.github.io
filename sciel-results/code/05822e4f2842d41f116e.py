import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound (LCB) for better exploration in low-budget BO
    # Minimization: we want to find low f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to select the point with the lowest LCB.
    # The acquisition function returns scores where HIGHER is better.
    # So we return -LCB(x) = -mu(x) + beta * sigma(x)
    
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
