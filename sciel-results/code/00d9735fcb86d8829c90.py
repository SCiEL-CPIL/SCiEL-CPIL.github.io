"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) inspired acquisition function with sublinear cost discount.
    
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
    
    # Probability that this candidate is the new minimum (approximation):
    # P(f(x) < f_best) = Phi(z)
    prob_new_min = norm.cdf(z)
    
    # Expected value if it's the new minimum:
    # E[f(x) | f(x) < f_best] = mu - sigma * pdf(z)/cdf(z)  (truncated normal mean)
    # But for KG, we want the expected improvement in the global optimum.
    # A common approximation for KG is:
    # KG(x) ≈ E[max(f_best - f(x), 0)] * P(f(x) < f_best) / f_best
    # However, a simpler and effective approach is to use the expected improvement
    # weighted by the probability of being the new minimum, discounted by cost.
    
    # Expected improvement (EI)
    ei = (f_best - mu) * prob_new_min + sigma_safe * norm.pdf(z)
    
    # KG-like score: expected improvement in the global optimum
    # We approximate the KG as the expected reduction in the global optimum value.
    # A practical approximation: weight EI by the probability that this point
    # is significantly better than the current best, and discount by cost.
    
    # Sublinear cost discount: 1/sqrt(cost)
    # This balances exploration (high uncertainty) against budget efficiency.
    cost_safe = np.maximum(cost, 1e-9)
    cost_discount = 1.0 / np.sqrt(cost_safe)
    
    # KG-inspired score:
    # Combine EI with a term that favors points likely to be new minima
    # and discount by cost using a sublinear function.
    # The probability of being a new minimum provides an exploration bonus.
    
    # Final score: EI * cost_discount * (1 + exploration_bonus)
    # Where exploration_bonus is based on the probability of being a new minimum
    scores = ei * cost_discount * (1.0 + prob_new_min)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
