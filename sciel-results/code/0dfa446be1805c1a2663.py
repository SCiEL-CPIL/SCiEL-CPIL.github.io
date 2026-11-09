"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Shekel 4d.
    Combines Expected Improvement (EI) with an uncertainty-weighted exploration
    bonus, normalized by cost to account for the varying evaluation cost.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Uncertainty-weighted exploration bonus (UCB-like)
    # Scale exploration based on remaining budget relative to total
    # We don't know total budget directly, but we can use cost as a proxy.
    # Lower cost = cheaper evaluation = more exploratory we can afford?
    # Actually, we want to balance exploitation (EI) and exploration (sigma).
    # A common blend is EI + beta * sigma, where beta is tuned.
    # Given the tight budget and multimodal nature, a moderate exploration bonus helps.
    
    # Adaptive beta: higher exploration when we are early (low cumulative cost) or when sigma is high.
    # Since we don't have access to the iteration count or total cost history directly in this function,
    # we rely on the fact that 'cost' is provided per candidate.
    # Let's use a fixed beta that was found to work well in similar scenarios,
    # or a simple heuristic based on the relative scale of mu and sigma.
    
    # A robust approach: use a small constant beta for UCB term.
    # The inspiration 1 (score 1.2425) likely used a specific blend.
    # Let's try EI + 0.1 * sigma, normalized by cost.
    # But wait, the cost is in the denominator. If we add sigma, it should also be scaled.
    
    # Let's refine:
    # Score = (EI + beta * sigma) / cost
    # beta = 0.1 seems reasonable for [0, 1] scaled functions.
    
    beta = 0.1
    score = (ei + beta * sigma_safe) / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
