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


def _probability_of_improvement(mu, sigma, f_best, xi=0.0):
    """Probability of improvement for minimization."""
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma
    pi = norm.cdf(z)
    return np.clip(pi, 0.0, 1.0)


def _qei(mu, sigma, f_best, xi=0.0):
    """Quantile-based EI with exploration parameter."""
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    # Use a slightly optimistic target to encourage exploration
    target = f_best - xi
    z = (target - mu) / sigma
    ei = (target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
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
    
    # Compute Probability of Improvement
    pi = _probability_of_improvement(mu, sigma, f_best, xi=0.0)
    
    # Compute QEI with small exploration parameter
    qei = _qei(mu, sigma, f_best, xi=0.005)
    
    # Cost-aware scoring: maximize improvement per unit cost
    # EI / cost directly optimizes marginal return on budget
    score_ei = ei / cost
    score_pi = pi / cost
    score_qei = qei / cost
    
    # Blend: primarily EI/cost, with PI/cost for stability and QEI/cost for exploration
    # Weights sum to 1.0
    w_ei = 0.60
    w_pi = 0.25
    w_qei = 0.15
    
    score = w_ei * score_ei + w_pi * score_pi + w_qei * score_qei
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Add a small random tiebreaker to avoid always picking the same point
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    tiebreaker = rng.uniform(0, 1e-8, size=len(score))
    score = score + tiebreaker
    
    return score
# EVOLVE-BLOCK-END
