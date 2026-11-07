import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero or invalid z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Soft-thresholding transform for Bayesian expected utility
    # This converts continuous improvement into a bounded probability-like signal
    # that is less sensitive to outlier GP predictions than raw EI or linear PI
    
    # Compute the improvement potential: how much better could this point be?
    improvement = np.maximum(0, f_best - mu)
    
    # Apply soft-thresholding transform with lambda=5.0
    # This creates a smooth, bounded signal that saturates for large improvements
    lambda_param = 5.0
    transformed_improvement = 1.0 - np.exp(-lambda_param * improvement)
    
    # Cost-efficiency scoring: divide by evaluation cost
    # This prioritizes points that offer good improvement per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = transformed_improvement / cost_safe
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
