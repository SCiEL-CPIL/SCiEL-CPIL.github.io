import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (EI) acquisition function.
    
    Parameters:
    mu: array of shape (n,), GP mean predictions
    sigma: array of shape (n,), GP standard deviations
    f_best: float, best observed function value so far
    cost: array of shape (n,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (n,), higher is better
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Z-score
    # Z = (f_best - mu) / sigma
    # If sigma is very small, Z can be huge.
    Z = (f_best - mu) / sigma
    
    # Calculate Phi(Z) and phi(Z)
    # scipy.stats.norm.cdf and norm.pdf handle large values well, 
    # but we should be careful with overflow/underflow if Z is extreme.
    # norm.cdf(Z) -> 1 for large positive Z, 0 for large negative Z.
    # norm.pdf(Z) -> 0 for large |Z|.
    
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: If f_best - mu is negative (mu > f_best), EI is technically 0 
    # in the standard definition if we assume we only care about improvement.
    # The formula above yields negative values if mu > f_best.
    # Standard EI is max(0, ...).
    # Let's calculate the raw term and then clamp to 0.
    EI_raw = (f_best - mu) * Phi_Z + sigma * phi_Z
    EI = np.maximum(EI_raw, 0.0)
    
    # Cost adjustment
    # We want to maximize improvement per unit cost.
    scores = EI / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
