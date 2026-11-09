import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Clip sigma to avoid numerical issues
        sigma = np.maximum(sigma, 1e-6)
        
        # Calculate EI
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Calculate UCB (for minimization, we want low mu, high sigma)
        # Score should be higher for better candidates.
        # UCB value = mu - beta * sigma. We want to minimize this.
        # So score = -UCB / cost = (beta * sigma - mu) / cost
        beta = 2.5 
        ucb_score = (beta * sigma - mu) / cost
        
        # Let's try a weighted combination or just pick one.
        # EI is standard. UCB is more exploratory.
        # Given the "Styblinski Tang" is multimodal, UCB might be better.
        # Let's return UCB score.
        
        # Ensure finite
        scores = ucb_score
        scores[~np.isfinite(scores)] = -np.inf
        
        return scores
