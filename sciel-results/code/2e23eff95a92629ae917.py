import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
        # Deadline guard
        # The problem states 330s total. This function is called repeatedly.
        # We can't easily track total time across calls without global state,
        # but we can ensure this call is fast.
        # The calculation is vectorized and fast.
        
        # Use UCB for minimization
        beta = 1.5
        # score = (beta * sigma - mu) / cost
        
        # Alternatively, stick to EI but maybe adjust?
        # Let's try the UCB approach.
        
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Let's try a blend: 0.5 * EI + 0.5 * (beta * sigma - mu) ?
        # No, units might not match perfectly if f_best is far from mu.
        # EI is in units of improvement. UCB term is in units of function value.
        # They are comparable.
        
        # Let's try pure UCB first. It's a distinct strategy from the seed.
        score = (beta * sigma - mu) / cost
        
        # Ensure finite
        score = np.where(np.isfinite(score), score, 0.0)
        
        return score
