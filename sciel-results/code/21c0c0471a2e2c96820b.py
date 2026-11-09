import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta for the Lower Confidence Bound
    # Scale the exploration term with the current best observed value
    # This ensures the acquisition function adapts to the magnitude of the objective
    beta = max(1.0, 0.1 * f_best)
    
    # Cost-Adjusted LCB: Maximize (beta * sigma - mu) / cost
    # This encourages exploring regions with high uncertainty (sigma) and low predicted mean (mu),
    # while penalizing regions with high evaluation cost.
    score = (beta * sigma - mu) / cost
    
    # Replace non-finite values with a large negative number to ensure validity
    score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return score
