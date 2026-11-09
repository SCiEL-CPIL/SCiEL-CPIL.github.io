"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a model-and-optimize approach for UCB.
    
    1. Defines a family of UCB acquisition functions parameterized by beta.
    2. Optimizes beta by selecting the value that maximizes the Expected Improvement (EI)
       of the best candidate identified by that UCB variant.
    3. Returns the scores for the optimal beta.
    """
    # Ensure sigma and cost are positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Define a grid of beta values to test
    # Beta controls the balance between exploitation (-mu) and exploration (beta * sigma)
    betas = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    best_beta = 2.0
    best_ei = -np.inf
    best_scores = None
    
    # Precompute terms that don't depend on beta
    # For minimization, potential improvement is f_best - mu
    # Z-score for EI calculation
    # We calculate EI for the candidate selected by each beta
    
    # To be efficient, we can compute scores for all betas at once if memory allows,
    # or iterate. 2000 points * 6 betas is small, so iteration is fine and clear.
    
    for beta in betas:
        # UCB score: -mu + beta * sigma
        ucb = -mu + beta * sigma
        # Cost-aware score
        scores = ucb / cost
        
        # Handle non-finite scores
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        # Find the best candidate according to this beta
        idx_max = np.argmax(scores)
        
        # Calculate Expected Improvement (EI) for this candidate
        # EI is the expected reduction in the objective function value
        # EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
        # where Z = (f_best - mu(x)) / sigma(x)
        
        mu_x = mu[idx_max]
        sigma_x = sigma[idx_max]
        
        if sigma_x < 1e-9:
            # If sigma is tiny, EI is just max(f_best - mu, 0)
            ei = max(f_best - mu_x, 0.0)
        else:
            z = (f_best - mu_x) / sigma_x
            # Use scipy.stats.norm for Phi and phi
            # Phi(z) = norm.cdf(z)
            # phi(z) = norm.pdf(z)
            ei = (f_best - mu_x) * norm.cdf(z) + sigma_x * norm.pdf(z)
            
        # We want to maximize the EI of the selected point
        # Note: EI is always non-negative. Higher is better.
        if ei > best_ei:
            best_ei = ei
            best_beta = beta
            best_scores = scores
            
    # If for some reason best_scores is None (should not happen), fallback
    if best_scores is None:
        ucb = -mu + best_beta * sigma
        best_scores = ucb / cost
        best_scores = np.where(np.isfinite(best_scores), best_scores, 0.0)
        
    # Ensure the returned scores are finite for the final output
    # The scores used for selection were -inf for invalids, but we return 0.0 for safety
    # as per task requirement "Return 2000 finite scores"
    return np.where(np.isfinite(best_scores), best_scores, 0.0)
# EVOLVE-BLOCK-END
