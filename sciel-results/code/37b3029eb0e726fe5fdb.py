import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # Cost-aware UCB for minimization
        # We want to find points with low expected value and high uncertainty.
        # LCB = mu - kappa * sigma
        # We want to minimize LCB, so we maximize -LCB = kappa * sigma - mu.
        # Divide by cost to account for evaluation cost.
        
        kappa = 2.5
        
        # Ensure sigma is strictly positive to avoid division issues or zero variance problems
        sigma = np.maximum(sigma, 1e-6)
        
        # Calculate the acquisition score
        score = (kappa * sigma - mu) / cost
        
        # Ensure the output is finite
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return score
