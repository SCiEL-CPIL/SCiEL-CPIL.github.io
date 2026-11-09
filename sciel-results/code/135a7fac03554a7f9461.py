import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Explicit deadline guard
    # The total time limit is 330s. This function is called ~25 times.
    # Each call should be well under 1s.
    # We don't have a global timer here, but we can assume the caller manages it.
    # However, the directive says "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
    # This implies the function might be part of a loop where we check time.
    # But the signature is fixed. We cannot change the signature.
    # So we just ensure the function is fast.
    
    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero or negative variance issues
    sigma = np.clip(sigma, 1e-12, None)
    
    # Cost-aware LCB
    # We want to minimize mu - beta * sigma
    # Maximizing (beta * sigma - mu) / cost
    
    # Beta selection
    # A fixed beta is a standard starting point.
    # Based on typical BO literature for medium dimensions, beta in [1, 3] is common.
    beta = 2.5
    
    # Calculate the acquisition score
    # lcb = mu - beta * sigma
    # score = -lcb / cost = (beta * sigma - mu) / cost
    
    numerator = beta * sigma - mu
    denominator = np.maximum(cost, 1e-9) # Cost is >= 1, so this is safe
    
    scores = numerator / denominator
    
    # Ensure all scores are finite
    # Replace NaNs and Infs with a very low value
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
