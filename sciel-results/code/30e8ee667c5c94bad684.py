"""Rewritten program using a different acquisition approach: UCB with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of UCB and normalized improvement.
    
    We use a UCB-like term that balances exploitation (low mu) and exploration (high sigma),
    then normalize by cost to account for evaluation expense.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions at candidate points.
    sigma : array of shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values are more desirable to evaluate next.
    """
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB exploration parameter: adapt based on how close we are to the optimum
    # As f_best approaches 0, we should explore more
    # Use a beta that increases as f_best gets smaller
    beta = 2.0 * np.sqrt(np.log(1.0 + np.maximum(1.0, f_best)))
    
    # UCB score: lower is better for minimization, so we negate
    # ucb = -mu + beta * sigma
    # But we want higher acquisition scores to be better, so:
    # acquisition = (f_best - mu + beta * sigma) / cost
    # This combines expected improvement with exploration
    
    # Alternatively, use a pure UCB-style:
    # We want to find points with low mu and/or high sigma
    # Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Let's use a combination: 
    # The expected improvement per cost is good, but let's try UCB per cost
    # UCB value (to minimize): mu - beta * sigma
    # Acquisition (to maximize): (f_best - (mu - beta * sigma)) / cost
    #                             = (f_best - mu + beta * sigma) / cost
    
    # However, this can be dominated by high-sigma regions early on.
    # Let's normalize sigma by its scale relative to the problem.
    
    # A better approach: use the ratio of expected improvement to cost,
    # but with a UCB twist.
    
    # Let's try: score = ((f_best - mu) * cdf(z) + sigma * pdf(z) + beta * sigma) / cost
    # This adds an exploration bonus.
    
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Avoid numerical issues
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration term
    exploration = beta * sigma
    
    # Combine EI and exploration, then divide by cost
    raw_score = ei + exploration
    
    # Normalize by cost
    scores = raw_score / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
