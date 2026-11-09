import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive exploration.
    
    Parameters:
    mu: array of predicted means (shape N,)
    sigma: array of predicted standard deviations (shape N,)
    f_best: best observed function value so far
    
    Returns:
    scores: array of scores (shape N,), higher is better
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Avoid division by zero in Z calculation
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Standardized distance from best
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Where Phi is CDF and phi is PDF of standard normal
    
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # To ensure we explore sufficiently early or if stuck, 
    # we can add a small exploration term based on sigma, 
    # but pure EI is often superior if f_best is updated correctly.
    # However, in the very first steps, f_best might be high.
    # A common trick is to use EI with a small xi (exploration parameter)
    # EI(xi) = (f_best - xi - mu) * Phi(Z') + sigma * phi(Z')
    # where Z' = (f_best - xi - mu) / sigma
    
    # Let's use a small xi to encourage exploration of points that are 
    # significantly better than the current best, but also allow some 
    # exploration of high uncertainty regions if the improvement is marginal.
    # Actually, standard EI (xi=0) is robust. 
    # The key difference from UCB is that EI naturally decays the exploration 
    # bonus as the probability of improvement drops.
    
    # To be safe against local optima in Goldstein-Price (which has many),
    # we can blend EI with a small amount of UCB-like exploration, 
    # or simply use EI which is theoretically grounded.
    # Given the "inspiration" that adding exploration to EI helped, 
    # and the current UCB score is 1.29, let's try a hybrid or pure EI.
    # Pure EI is often the gold standard. Let's stick to pure EI first.
    
    # Ensure finite scores
    scores = np.where(np.isfinite(EI), EI, 0.0)
    
    return scores
