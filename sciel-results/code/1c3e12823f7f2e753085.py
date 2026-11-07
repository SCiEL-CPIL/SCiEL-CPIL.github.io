import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a soft-thresholded improvement
    metric normalized by evaluation cost.
    
    Parameters:
    mu: np.ndarray, shape (2000,), GP mean predictions.
    sigma: np.ndarray, shape (2000,), GP standard deviation predictions.
    f_best: float, best observed function value.
    cost: np.ndarray, shape (2000,), cost of evaluating each candidate.
    
    Returns:
    scores: np.ndarray, shape (2000,), acquisition scores.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate the expected improvement (positive when mu < f_best)
    improvement = np.maximum(f_best - mu, 0.0)
    
    # Apply a soft-thresholding transform to bound the improvement signal
    # This makes the score less sensitive to outlier GP predictions
    # lambda controls the scale; larger lambda means sharper transition
    lam = 1.0
    soft_improvement = 1.0 - np.exp(-lam * improvement)
    
    # Multiply by uncertainty to balance exploration and exploitation
    # This creates a multiplicative balance that favors cheap, uncertain, promising regions
    uncertainty_weight = sigma_safe
    
    # Combine: soft improvement * uncertainty / cost
    scores = soft_improvement * uncertainty_weight / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
