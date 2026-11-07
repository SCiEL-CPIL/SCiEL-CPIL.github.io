import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 5.0 # 5 seconds safety margin
    
    # EI with xi = 1.0 to encourage significant improvements
    xi = 1.0
    z = (f_best - xi - mu) / (sigma + 1e-9)
    scores = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
