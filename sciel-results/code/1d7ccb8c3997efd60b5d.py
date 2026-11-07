"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) acquisition function with sublinear cost discount.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions for 2000 candidates.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidates.
    f_best : float
        Current best observed value (lowest).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: how much better than current best, in std units
    z = (f_best - mu) / sigma_safe
    
    # Probability that this candidate is the new minimum:
    # P(f(x) < f_best) = Phi(z)
    prob_new_min = norm.cdf(z)
    
    # Expected improvement (EI)
    ei = (f_best - mu) * prob_new_min + sigma_safe * norm.pdf(z)
    
    # KG approximation: expected reduction in global optimum
    # Approximate as EI weighted by probability of being new minimum
    # This gives more weight to points that are both likely to improve
    # and have high expected improvement
    kg_approx = ei * prob_new_min
    
    # Sublinear cost discount: 1/sqrt(cost)
    cost_safe = np.maximum(cost, 1e-9)
    cost_discount = 1.0 / np.sqrt(cost_safe)
    
    # Final score: KG approximation discounted by cost
    scores = kg_approx * cost_discount
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
