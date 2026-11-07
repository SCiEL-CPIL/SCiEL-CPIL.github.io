"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Instead of Expected Improvement, this uses an optimistic bound on the function value
    (or rather, a pessimistic bound on the minimum, since we are minimizing).
    
    For minimization, we want to find low values.
    The GP predicts mu (mean) and sigma (std dev).
    We want to explore regions where the true function value could be very low.
    The lower bound of the function value is roughly mu - beta * sigma.
    We want to maximize the probability of finding a low value, or maximize the "potential" for improvement.
    
    Standard UCB for minimization (to find the minimum):
    We want to select x that minimizes the lower bound: L(x) = mu(x) - beta * sigma(x).
    However, the acquisition function returns scores where HIGHER is better.
    So we should return -L(x) / cost = (beta * sigma - mu) / cost?
    Or, we can frame it as maximizing the "regret" bound.
    
    Let's stick to the standard form adapted for minimization.
    If we define the acquisition as the "potential improvement" over f_best:
    We expect the function to be at least mu - beta*sigma.
    If mu - beta*sigma < f_best, there is potential.
    
    A common cost-aware UCB for minimization is:
    Score = (beta * sigma - mu) / cost
    This rewards high sigma (exploration) and low mu (exploitation).
    
    Let's verify the sign.
    If mu is very low (good), score is high (if beta*sigma is not huge negative, which it isn't).
    If sigma is high (uncertain), score is high.
    If cost is high, score is low.
    
    This seems structurally different from EI. EI is 0 if mu > f_best (assuming no noise). UCB-like scores can be positive even if mu > f_best, if sigma is large enough. This drives exploration more aggressively.
    """
    # Ensure cost is at least 1 to avoid division issues, though contract says cost >= 1
    cost_safe = np.maximum(cost, 1e-6)
    
    # Exploration coefficient
    # A value of 3.0 is a standard heuristic for UCB in BO
    beta = 3.0
    
    # For minimization, we want to maximize the chance of finding a value lower than f_best.
    # The lower bound of the function is mu - beta * sigma.
    # We want this lower bound to be as small as possible.
    # So we maximize (beta * sigma - mu).
    # We divide by cost to account for evaluation expense.
    
    scores = (beta * sigma - mu) / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
