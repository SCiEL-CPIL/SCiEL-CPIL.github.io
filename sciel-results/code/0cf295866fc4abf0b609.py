"""Cost-aware Expected Improvement acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    EI directly measures the expected reduction in the best observed value,
    which is theoretically well-founded for minimization. We normalize by
    cost^0.8 to soften the cost penalty, allowing exploration of points with
    higher expected improvement even at slightly higher cost.
    
    The exploration is adaptively controlled: when f_best is high (early),
    we add a small uncertainty bonus to encourage exploration. When f_best
    is low (late), we rely more on exploitation.
    """
    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    # For minimization: improvement = f_best - f(x)
    # If f(x) ~ N(mu, s^2), then EI = E[max(0, f_best - f(x))]
    z = (f_best - mu) / s
    
    # Expected improvement formula
    # EI = (f_best - mu) * Phi(z) + s * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # When f_best is high (early), we want more exploration
    # Use a soft cap: exploration bonus decays as f_best improves
    # f_best for Ackley: starts around ~5-10, ends near 0
    # exploration_weight = exp(-0.3 * f_best) gives ~1.0 when f_best=0, ~0.05 when f_best=10
    # But we want MORE exploration when f_best is HIGH (bad), not low (good)
    # So: exploration_weight = 1 - exp(-0.3 * f_best)
    # When f_best = 0: weight = 0 (no exploration needed)
    # When f_best = 10: weight = 0.95 (lots of exploration)
    exploration_weight = 1.0 - np.exp(-0.3 * f_best)
    exploration_weight = np.clip(exploration_weight, 0.0, 1.0)
    
    # Uncertainty bonus: proportional to sigma, scaled by exploration weight
    # This encourages evaluating points where we are uncertain
    uncertainty_bonus = exploration_weight * 0.5 * s
    
    # Combined score before cost normalization
    score = ei + uncertainty_bonus
    
    # Cost awareness: divide by cost^0.8 (softened penalty)
    c = np.maximum(cost, 1e-9)
    scores = score / np.power(c, 0.8)
    
    # Safety: Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
