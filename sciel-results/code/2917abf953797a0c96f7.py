"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    mu: array of mean predictions, shape (N,)
    sigma: array of standard deviation predictions, shape (N,)
    f_best: float, the best (lowest) observed function value so far.
    
    Returns:
    scores: array of acquisition scores, shape (N,). Higher scores indicate more promising points.
    """
    # Exploration parameter for UCB
    # Tuned for 2D, 30 evals, multi-modal RBF mixture
    beta = 3.0
    
    # Sanitize inputs
    sigma = np.asarray(sigma)
    mu = np.asarray(mu)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Handle non-finite values
    # If mu is not finite, it's a bad prediction, set score to -inf
    # If sigma is not finite, set to 0
    mu_clean = np.where(np.isfinite(mu), mu, f_best)
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # UCB for minimization: 
    # We want to minimize f(x). 
    # The "upper bound" of the potential is mu + beta*sigma.
    # We want to pick the point with the LOWEST upper bound? 
    # No, UCB for minimization is usually defined as minimizing mu - beta*sigma.
    # Or maximizing (f_best - mu) + beta*sigma.
    # Let's verify:
    # If we minimize, we are looking for low values.
    # A point is attractive if its mean is low (mu << f_best) OR its uncertainty is high (sigma is large, might be even lower).
    # So we want to maximize: (f_best - mu) + beta * sigma.
    # This quantity is large when mu is small and sigma is large.
    
    scores = (f_best - mu_clean) + beta * sigma_clean
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
