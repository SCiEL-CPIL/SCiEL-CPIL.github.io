"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a budget-constrained UCB on the
    negative expected regret (reward) with sqrt(cost) normalization.
    
    Strategy:
    1. Compute reward as negative expected regret: r = -mu + beta * sigma
    2. Normalize by sqrt(cost) to account for evaluation cost
    3. Beta scales with remaining budget ratio to shift from exploration to exploitation
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(2000, dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Estimate remaining budget ratio
    # Total budget is 30, initial 5 points cost ~1.25 each = ~6.25
    # So remaining budget is approximately 30 - 6.25 = 23.75 at start
    # We don't know exact spend, so use f_best as proxy for progress
    # When f_best is large (near start), we have more budget relative to need
    # When f_best is small (near end), we need to conserve
    
    # Beta scales with how far we are from optimum
    # f_best ranges from ~5+ (start) to ~0 (optimum)
    f_best_abs = abs(f_best)
    
    # Beta: start high for exploration, decrease as we approach optimum
    # Use a smooth decay based on f_best
    beta = 1.0 + 2.0 / (1.0 + f_best_abs * 0.5)
    beta = np.clip(beta, 0.5, 4.0)
    
    # Reward: negative expected regret
    # For minimization, regret = max(0, mu - f_best) approximately
    # But we want to maximize improvement potential
    # reward = -mu + beta * sigma captures both exploitation (-mu) and exploration (beta*sigma)
    reward = -mu + beta * sigma_safe
    
    # Cost-aware normalization: divide by sqrt(cost)
    # This makes cheaper evaluations more attractive
    cost_safe = np.maximum(cost, 1.0)
    cost_norm = np.sqrt(cost_safe)
    
    score = reward / cost_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
