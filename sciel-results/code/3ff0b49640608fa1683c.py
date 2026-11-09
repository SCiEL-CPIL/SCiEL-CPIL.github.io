import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    For minimization, we maximize the expected improvement over f_best.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-10)
    
    # EI for minimization:
    # Let Y = f(x). We want E[max(f_best - Y, 0)].
    # If Y ~ N(mu, sigma^2), let Z = (f_best - mu) / sigma.
    # Then E[max(f_best - Y, 0)] = sigma * [ Z * Phi(Z) + phi(Z) ]
    
    # Calculate Z
    # Avoid division by zero, though sigma is maxed at 1e-10
    Z = (f_best - mu) / sigma
    
    # Calculate Phi(Z) and phi(Z)
    # norm.cdf is Phi, norm.pdf is phi
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    scores = sigma * (Z * Phi_Z + phi_Z)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 1.0:
        pass
        
    return scores
