"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We will use a hybrid of EI and UCB, normalized by cost, but with a
# time-deadline guard and a slightly different weighting to balance
# exploration and exploitation.

def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement."""
    z = (f_best - mu) / (sigma + 1e-12)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def _ucb(mu, sigma, f_best, beta=1.0):
    """Compute Upper Confidence Bound (for minimization, use lower confidence)."""
    # For minimization, we want to explore where the lower bound is low.
    # LCB = mu - beta * sigma
    # But the acquisition should be high when LCB is low, so we can use -LCB or similar.
    # Actually, standard UCB for minimization is: mu - beta*sigma (we want to minimize this).
    # The acquisition function should return high values for points we want to evaluate.
    # So if we want to evaluate points with low LCB, we can return -LCB = -mu + beta*sigma.
    # But we also need to incorporate f_best somehow? 
    # Let's stick with a combination.
    return -mu + beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    mu, sigma, cost: shape (2000,)
    f_best: scalar
    Returns: shape (2000,) finite scores, higher = evaluate next.
    """
    start_time = time.time()
    deadline = 300.0  # seconds, leave margin for safety

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)

    # Compute EI
    ei = _expected_improvement(mu, sigma, f_best)

    # Compute a UCB-like term for minimization
    # We want to explore areas with high uncertainty.
    # A common approach is to use a weighted combination.
    # Let's try: score = (alpha * EI + (1 - alpha) * UCB_term) / cost
    # UCB_term for minimization: we can use -mu + beta*sigma, but we need to scale it.
    # Alternatively, we can use a "probability of improvement" or a different metric.
    # Let's try a simpler blend: EI/cost and a term that rewards high sigma.
    
    # Normalize EI and sigma to have comparable scales?
    # Or just use a fixed blend.
    
    # Let's try: score = (EI + beta * sigma) / cost
    # This encourages both high expected improvement and high uncertainty.
    
    # But EI already includes sigma. 
    # Let's try a different approach: use a "knowledge gradient" or "probability of best" approximation.
    # For now, let's stick with a blend of EI and a UCB-like term.
    
    # UCB for minimization: we want to minimize mu - beta*sigma.
    # So the "desirability" is high when mu - beta*sigma is low.
    # We can use: desirability_ucb = -(mu - beta*sigma) = -mu + beta*sigma
    # But this doesn't incorporate f_best. 
    # Let's scale it by f_best or use a relative term.
    
    # A better idea: use the "expected improvement" and "uncertainty" separately.
    # score = (w_ei * EI + w_ucb * (beta * sigma)) / cost
    # where w_ei + w_ucb = 1.
    
    # Let's try w_ei = 0.7, w_ucb = 0.3, beta = 1.0.
    
    w_ei = 0.7
    w_ucb = 0.3
    beta = 1.0
    
    score = (w_ei * ei + w_ucb * beta * sigma) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return the best score found so far (which is score)
        return score
    
    return score
# EVOLVE-BLOCK-END
