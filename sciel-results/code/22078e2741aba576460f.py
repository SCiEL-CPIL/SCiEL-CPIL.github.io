import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
        # Deadline guard
        # The prompt implies a global wall-clock, but we only have local time.
        # We assume the function is called frequently.
        # We will just compute the score.
        
        # Avoid division by zero or negative costs (though cost >= 1)
        cost_safe = np.maximum(cost, 1e-6)
        
        # UCB for minimization: -mu + beta * sigma
        # We maximize this score.
        beta = 1.0
        
        score = (-mu + beta * sigma) / cost_safe
        
        # Ensure finite scores
        score = np.where(np.isfinite(score), score, -np.inf)
        
        return score
