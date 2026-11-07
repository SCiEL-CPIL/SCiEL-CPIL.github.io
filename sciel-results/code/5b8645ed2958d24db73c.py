import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Probability of Improvement (PI)
        # We want to maximize the probability that f(x) < f_best
        # f(x) ~ N(mu, sigma^2)
        # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
        
        # Add small epsilon to avoid division by zero
        sigma_safe = np.maximum(sigma, 1e-9)
        
        z = (f_best - mu) / sigma_safe
        scores = norm.cdf(z)
        
        # Ensure finite scores
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        return scores
