import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e6, neginf=1e-10)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-10)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    # 20 is a safe upper bound for double precision stability in cdf/pdf
    z = np.clip(z, -20, 20)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score: higher is better
    # We use Pure Cost-Aware EI. The exploration is inherently handled by the sigma term in EI.
    # Adding an extra linear sigma term can sometimes distort the acquisition landscape.
    scores = ei / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check time
    if time.time() - start_time > 290.0:
        return scores
    
    return scores
