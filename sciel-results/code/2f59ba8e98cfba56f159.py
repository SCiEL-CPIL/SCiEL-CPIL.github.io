"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement (EI) acquisition function for minimization.
    
    The standard EI for minimization is:
    EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (f_best - mu(x)) / sigma(x), Phi is CDF, phi is PDF of standard normal.
    
    To make it scale-free and robust, we use a normalized version where we divide by 
    a measure of the overall uncertainty, often approximated by the mean sigma or 
    max sigma, or simply use the raw EI which is already in objective units. 
    However, since mu and sigma are in objective units, raw EI is in objective units.
    
    The inspiration suggests a "dynamic, normalized hybrid". Let's implement a 
    variant that adapts the exploration based on the current state. 
    A common robust approach is to use the "Expected Improvement" directly, 
    but scaled to ensure it competes well with other terms if this were a hybrid. 
    Since we are doing pure EI here, we return the EI values. 
    Higher EI is better.
    
    To handle the "normalized" aspect mentioned in inspiration 1, we can normalize 
    the EI scores by the standard deviation of the EI scores themselves or by the 
    global uncertainty. However, simply returning EI is often strong. 
    Let's try a slightly modified EI that accounts for the fact that we want 
    to maximize the score.
    
    Let's stick to the standard EI formula but ensure numerical stability.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # For minimization, we expect improvement if mu < f_best.
    # The term (f_best - mu) is positive when mu < f_best.
    
    # Use scipy.stats.norm for precision
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip small negative values to 0 (numerical noise)
    EI = np.maximum(EI, 0.0)
    
    # The score should be higher for better candidates.
    # EI is already in objective units (lower is better for objective, so higher EI is better).
    
    # Ensure finite
    EI = np.nan_to_num(EI, nan=0.0, posinf=1e6, neginf=0.0)
    
    return EI
# EVOLVE-BLOCK-END
