import numpy as np

def acquisition(mu, sigma, f_best, cost):
       # Lower Confidence Bound (LCB) for minimization
       # We want to find points with low mu and high sigma
       # lcb = mu - kappa * sigma
       # We want to maximize the "desirability", so we use -lcb / cost
       # score = (kappa * sigma - mu) / cost
       
       kappa = 1.0
       
       # Ensure sigma is non-negative
       sigma = np.maximum(sigma, 0)
       
       # Calculate score
       numerator = kappa * sigma - mu
       cost_safe = np.maximum(cost, 1e-9)
       scores = numerator / cost_safe
       
       # Handle non-finite values
       scores = np.where(np.isfinite(scores), scores, 0.0)
       
       return scores
