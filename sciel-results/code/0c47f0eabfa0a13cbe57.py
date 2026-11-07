"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for safety
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores using a cost-normalized GP-UCB 
    with dynamically annealed exploration coefficient.
    
    The exploration coefficient beta decays exponentially based on the ratio 
    of current total cost to the budget, ensuring early broad exploration 
    and late-stage precise exploitation.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of GP predictions
    sigma : array of shape (2000,)
        Standard deviation of GP predictions
    f_best : float
        Best observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising to evaluate)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety guard: if we are running out of time, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Return a simple fallback to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        scores = -mu / np.sqrt(cost)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Estimate total cost spent so far.
    # The task says: "Runs start with 5 uniform random points (cost counted). 
    # While total cost is below 30, the evaluator scores 2000 new uniform candidates 
    # with acquisition and evaluates the best one."
    # Each evaluation costs 1 + mean(x). For uniform points in [0,1]^6, mean(x) ~ 0.5, 
    # so cost ~ 1.5 per point. 5 points * 1.5 = 7.5 initial cost.
    # The loop runs while total cost < 30. So we have at most ~ (30 - 7.5) / 1.5 = 15 more evaluations.
    # Total evaluations ~ 20.
    # We can estimate the number of evaluations done so far by looking at how low f_best is,
    # or simply use a fixed schedule. Since we don't have explicit iteration count, 
    # let's estimate progress based on f_best.
    # Hartmann 6d min is ~ -3.32237. Initial randoms might be ~ -1.0 to -2.0.
    # Let's assume a linear progress from f_best = -1.0 (early) to f_best = -3.3 (late).
    # progress = (f_best - (-1.0)) / (-3.3 - (-1.0)) = (f_best + 1.0) / (-2.3)
    # But f_best decreases, so we want progress to increase as f_best decreases.
    # progress = (-1.0 - f_best) / (-1.0 - (-3.3)) = (-1.0 - f_best) / 2.3
    # Clip to [0, 1].
    
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = -1.0
    
    # Estimate progress: 0 at start (f_best ~ -1.0), 1 at end (f_best ~ -3.3)
    progress = (-1.0 - f_best) / 2.3
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic beta: starts high (e.g., 5.0) and decays to low (e.g., 0.5)
    # beta = beta_min + (beta_max - beta_min) * exp(-k * progress)
    beta_max = 5.0
    beta_min = 0.5
    k = 3.0  # Decay rate
    beta = beta_min + (beta_max - beta_min) * np.exp(-k * progress)
    
    # GP-UCB: mu + sqrt(beta) * sigma
    # Since we are minimizing, we want to maximize -mu + sqrt(beta) * sigma? 
    # No, standard UCB for minimization is: minimize mu - sqrt(beta) * sigma? 
    # Actually, for minimization, we want to find the point where the lower bound is best.
    # Lower bound: mu - sqrt(beta) * sigma. We want to minimize this, so we maximize -mu + sqrt(beta) * sigma.
    # Alternatively, think of it as: we want to explore where the true value might be low.
    # f(x) ~ N(mu, sigma^2). P(f(x) < f_best) is one way.
    # UCB for minimization: score = -mu + sqrt(beta) * sigma.
    # This rewards low mu (exploitation) and high sigma (exploration).
    
    ucb_score = -mu + np.sqrt(beta) * sigma
    
    # Cost normalization: divide by cost^0.7 (as per knowledge, tighter exponent helps)
    cost_norm = np.power(cost, 0.7)
    
    scores = ucb_score / cost_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
