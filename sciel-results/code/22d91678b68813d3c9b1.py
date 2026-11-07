import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Strategy: Cost-Weighted Expected Improvement (CW-EI).
    
    Expected Improvement (EI) is a standard acquisition function for Bayesian Optimization 
    that balances exploration and exploitation by considering both the mean prediction 
    and the uncertainty. It is generally superior to Probability of Improvement (PI) 
    because it takes into account the *magnitude* of the potential improvement, not just
    the probability.
    
    The cost-aware aspect is introduced by dividing the EI by the evaluation cost. 
    This transforms the objective to maximize "Expected Improvement per Unit Cost".
    
    Formula:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    and Phi is the standard normal CDF, phi is the standard normal PDF.
    
    Score(x) = EI(x) / cost(x)
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety checks
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0) # Min cost is 1.0
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Cost-weighted score
    score = ei / cost
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback: simple exploitation based on cost-adjusted mean
        score = -mu / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
