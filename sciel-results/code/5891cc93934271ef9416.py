import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Calculate standard Expected Improvement
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Entropy-weighted EI: Amplify scores in high-uncertainty regions
        # This acts as a structural change to the exploration-exploitation tradeoff
        # by making the acquisition function super-linear in sigma.
        score = ei * (sigma + 1e-6)
        
        return score
