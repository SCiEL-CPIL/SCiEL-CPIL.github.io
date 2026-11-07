"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Pareto-frontier inspired scoring.
    We compute a score for each candidate based on the marginal reduction in
    predicted value per unit cost, but with a structural twist: we use a
    non-dominated sort concept where we rank candidates by their efficiency
    and assign scores based on their position on the frontier.
    
    However, since we need to return a scalar score for each of the 2000 candidates
    and the evaluator picks the best one, we can use a simpler but effective approach:
    score = expected_improvement / cost, but we add a small exploration bonus
    based on the uncertainty (sigma) to encourage exploration in early stages.
    
    The key insight from the knowledge is that cost normalization via division
    and stochastic exploration (Thompson Sampling) can help escape local optima.
    Since we can't do full Thompson Sampling in the acquisition function (we need
    deterministic scores), we add a small sigma-based bonus to encourage exploration.
    """
    # Add small epsilon to avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # Add a small exploration bonus proportional to sigma to encourage exploration
    # This helps escape local optima, especially in early iterations
    exploration_bonus = 0.1 * sigma
    
    # Final score: EI per unit cost plus exploration bonus per unit cost
    score = (ei + exploration_bonus) / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
