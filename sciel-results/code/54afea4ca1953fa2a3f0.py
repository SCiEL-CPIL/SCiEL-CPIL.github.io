"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and a stochastic Thompson-Sampling-like perturbation.
    
    The core idea is to use EI as the primary signal (which is robust and
    well-calibrated) but add a small stochastic component to encourage
    exploration in high-uncertainty regions, especially when the cost is low.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each point (1 + mean(x) for x in [0,1]^6).
    
    Returns:
    --------
    scores : array_like, shape (2000,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-adjusted EI
    # We want to balance improvement against cost.
    # A simple approach is EI / cost, but this can be too aggressive.
    # Instead, we use a power of cost to soften the penalty.
    cost_adj = ei / (cost ** 0.5)
    
    # Add a small stochastic exploration term based on sigma
    # This helps escape local optima and explore high-uncertainty regions
    # The scale of exploration is reduced as we get more certain (sigma -> 0)
    # and scaled by cost to avoid exploring expensive points too much
    
    # Use a fixed random seed for reproducibility within a single call
    # (though in practice, the evaluator may call this multiple times)
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(mu))
    
    # Exploration term: proportional to sigma, but damped by cost
    # The factor 0.1 controls the strength of exploration
    exploration = 0.1 * sigma_safe * noise / (cost ** 0.25)
    
    # Combine exploitation and exploration
    scores = cost_adj + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
