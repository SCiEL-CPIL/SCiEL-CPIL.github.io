import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # Cost-aware UCB for minimization
        # We want to maximize the score.
        # Potential value: -mu (lower is better) + kappa * sigma (uncertainty)
        kappa = 1.0
        ucb = -mu + kappa * sigma
        scores = ucb / cost
        
        # Handle non-finite values
        scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        
        return scores
