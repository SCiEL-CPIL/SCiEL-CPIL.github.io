"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 320.0  # seconds, safety margin before 330s kill
_T0 = time.time()


def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard expected improvement for minimization."""
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, beta=2.5):
    """Upper confidence bound for minimization (lower is better)."""
    # For minimization, we want to maximize the potential improvement.
    # Low mu is good. High sigma is good (uncertainty).
    # So we maximize (mu - beta * sigma) ? No, that would favor high mu.
    # We want to find the minimum.
    # The "Upper Confidence Bound" for the *regret* or *improvement*?
    # Standard UCB for minimization: minimize mu + beta * sigma.
    # Acquisition function should be HIGH where we want to go.
    # So we want HIGH where mu is LOW and sigma is HIGH.
    # Score = -mu + beta * sigma.
    # Or equivalently, maximize ( - (mu - beta * sigma) )? No.
    # Let's define UCB_score = beta * sigma - mu.
    # If mu is low (good) and sigma is high (uncertain), score is high.
    return beta * sigma - mu


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Max(EI/cost, UCB/cost).
    
    Parameters:
    - mu: (2000,) mean predictions from GP
    - sigma: (2000,) standard deviation predictions from GP
    - f_best: lowest observed value
    - cost: (2000,) cost of evaluating each point
    
    Returns:
    - (2000,) acquisition scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() - _T0 > _DEADLINE:
        return np.zeros(len(mu))
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best, xi=0.0)
    score_ei = ei / cost
    
    # Compute UCB-based score
    # beta=2.5 provides a balance between exploitation and exploration
    ucb_score = _ucb(mu, sigma, beta=2.5)
    # Normalize UCB score to be comparable to EI?
    # EI is roughly O(sigma). UCB is O(sigma).
    # However, EI is always positive. UCB can be negative.
    # We want to maximize the score.
    # If UCB is negative, it means mu is very high (bad) and sigma is not high enough to compensate.
    # We should probably clip UCB score at 0 or just take max with EI.
    # Since EI is >= 0, max(EI, UCB) will naturally ignore bad UCBs if EI is 0?
    # If EI is 0 and UCB is negative, max is negative.
    # We should ensure scores are non-negative if possible, or just relative ranking matters.
    # The evaluator picks the argmax. Negative scores are fine as long as one is higher.
    # However, to be safe and consistent with "higher is better", we can shift UCB.
    # But max(EI/cost, UCB/cost) is a valid structural choice.
    
    score_ucb = ucb_score / cost
    
    # Structural Change: Max-Combiner
    # This allows the algorithm to switch modes:
    # If EI is dominant (exploit), it picks the best exploitation point.
    # If UCB is dominant (explore), it picks the best exploration point.
    # Both are weighted by cost.
    
    score = np.maximum(score_ei, score_ucb)
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Add a small random tiebreaker to avoid always picking the same point
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    tiebreaker = rng.uniform(0, 1e-8, size=len(score))
    score = score + tiebreaker
    
    return score
# EVOLVE-BLOCK-END
