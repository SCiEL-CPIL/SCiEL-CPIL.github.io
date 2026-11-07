import numpy as np

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta: 
    # We scale the exploration term based on the current best value relative to the typical uncertainty.
    # This allows the algorithm to be more exploratory when the best value is far from the optimum (large f_best)
    # and more exploitative when the best value is close to the optimum (small f_best).
    
    # Use median sigma to be robust to outliers (unexplored regions might have high sigma)
    med_sigma = np.median(sigma)
    
    # Avoid division by zero
    if med_sigma < 1e-6:
        med_sigma = 1e-6
        
    # Beta scales with the potential for improvement (f_best) relative to uncertainty
    # We add a small floor to beta to ensure some exploration always happens
    beta = max(f_best / med_sigma, 0.5)
    
    # Cap beta to prevent excessive exploration if f_best is very large (shouldn't happen much)
    beta = min(beta, 5.0)
    
    # UCB for minimization: maximize -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
