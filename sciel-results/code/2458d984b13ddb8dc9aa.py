import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Handle numerical stability
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        
        # UCB for minimization
        # We want to find low values.
        # Score should be high for low mu and high sigma.
        # Score = (-mu + kappa * sigma) / cost
        kappa = 2.0
        
        score = (-mu + kappa * sigma) / cost
        
        # Ensure finite output
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return score
