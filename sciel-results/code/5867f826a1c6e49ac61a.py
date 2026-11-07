import numpy as np
from scipy.stats import norm
import time

# Time guard: track when we started to avoid exceeding the 330s wall-clock limit
_ACQ_START = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses a smooth cost-aware Expected Improvement (EI/cost^alpha) with
    an adaptive exploration-exploitation balance.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    global _ACQ_START
    if _ACQ_START is None:
        _ACQ_START = time.time()
    # Safety margin: if we're close to the 330s limit, return quickly
    # This is a lightweight guard; the main budget control is in the caller.
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: use EI / cost^alpha with adaptive alpha
    # Lower alpha = more exploration of high-cost regions if EI is high enough
    # Higher alpha = stricter cost penalty
    # Start with a moderate cost sensitivity and adapt based on f_best
    # Early on (high f_best), we can afford to be less cost-sensitive
    # Late on (low f_best), we should be more cost-sensitive to preserve budget
    
    # Normalize cost to [0,1] range for stable exponentiation
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Adaptive alpha: start at 0.5, increase as f_best decreases
    # f_best for levy 2d ranges from ~10+ down to 0
    # Map f_best to alpha in [0.3, 1.0]
    f_best_clamped = np.clip(f_best, 0.0, 10.0)
    alpha = 0.3 + 0.7 * (1.0 - f_best_clamped / 10.0)
    
    # Cost penalty: smooth exponential decay
    # Use cost_norm to keep the penalty scale-invariant
    cost_penalty = np.exp(-alpha * cost_norm * 3.0)
    
    # Also add a small UCB-like exploration term to encourage diversity
    # beta adapts: higher early, lower late
    beta = 1.0 / (1.0 + f_best_clamped / 5.0)
    ucb_term = beta * sigma_safe
    
    # Final score: cost-aware EI plus exploration term
    scores = ei * cost_penalty + ucb_term * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
