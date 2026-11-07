"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 320.0  # seconds, safety margin before 330s kill


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


def _ucb_score(mu, sigma, beta=2.0):
    """UCB score for minimization: lower mu - beta*sigma is better.
    Returns a score where higher is better by normalizing."""
    ucb = mu - beta * sigma
    min_ucb = np.min(ucb)
    return (min_ucb - ucb)  # higher score = better (closer to min)


def _exploit_score(mu):
    """Pure exploitation: lowest predicted value is best.
    Returns score where higher is better."""
    min_mu = np.min(mu)
    return (min_mu - mu)  # higher score = better


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of EI, PI, UCB, and exploitation
    strategies, all normalized by cost (EI/cost approach).
    
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
    
    # Compute multiple acquisition strategies
    
    # 1. Expected Improvement per unit cost
    ei = _expected_improvement(mu, sigma, f_best, xi=0.0)
    score_ei = ei / cost
    
    # 2. Probability of Improvement per unit cost
    pi = _probability_of_improvement(mu, sigma, f_best, xi=0.0)
    score_pi = pi / cost
    
    # 3. UCB score per unit cost (exploration-exploitation balance)
    score_ucb = _ucb_score(mu, sigma, beta=2.0) / cost
    
    # 4. Pure exploitation per unit cost
    score_exploit = _exploit_score(mu) / cost
    
    # 5. Exploration: high uncertainty per cost (small weight)
    score_explore = sigma / cost
    
    # Blend strategies with weights that favor exploitation and EI
    # EI is the primary signal, PI provides stability, UCB balances exploration
    w_ei = 0.50
    w_pi = 0.20
    w_ucb = 0.15
    w_exploit = 0.10
    w_explore = 0.05
    
    score = (w_ei * score_ei + 
             w_pi * score_pi + 
             w_ucb * score_ucb + 
             w_exploit * score_exploit + 
             w_explore * score_explore)
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Add a small random tiebreaker to avoid always picking the same point
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    tiebreaker = rng.uniform(0, 1e-8, size=len(score))
    score = score + tiebreaker
    
    return score
# EVOLVE-BLOCK-END
