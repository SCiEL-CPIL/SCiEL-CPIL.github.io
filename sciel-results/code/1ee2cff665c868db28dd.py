import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330 - 10  # 320s to be safe

def acquisition(mu, sigma, f_best, cost):
    if time.time() > _DEADLINE:
        # Fallback: simple EI/cost
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        return np.maximum((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z), 0) / cost
    
    eps = 1e-9
    
    # Structural Change: Pure UCB with Cost Normalization
    # Instead of blending EI and UCB, we rely solely on the optimistic bound.
    # This maximizes the potential improvement assuming the worst-case uncertainty.
    
    # Beta parameter: Fixed at 2.5 for strong exploration in small-sample regime.
    # This is a structural choice: constant exploration pressure vs adaptive.
    beta = 2.5
    
    # UCB for minimization: We want to maximize (f_best - mu + beta * sigma)
    # This is equivalent to minimizing (mu - beta * sigma)
    ucb_imp = f_best - mu + beta * sigma
    
    # Ensure non-negative improvement score to avoid preferring points that are predicted to be worse
    # even with uncertainty, unless the uncertainty is huge (handled by beta)
    # Actually, UCB can be negative if mu > f_best + beta*sigma.
    # We should clip to 0 to ensure we don't pick points that are definitely worse.
    ucb_imp = np.maximum(ucb_imp, 0)
    
    # Cost-aware: Divide by cost to maximize improvement per unit cost
    score = ucb_imp / (cost + eps)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
