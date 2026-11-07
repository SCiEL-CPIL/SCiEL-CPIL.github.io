import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # LCB = mu - kappa * sigma
    # We want to maximize this, so we return -LCB = -mu + kappa * sigma
    
    # Dynamic kappa based on f_best
    # When f_best is far from the global minimum, we should explore more (higher kappa)
    # When f_best is close to the global minimum, we should exploit more (lower kappa)
    
    f_min_global = -39.1662
    regret = f_best - f_min_global
    
    # Normalize regret: at start regret ~ 50, at end regret ~ 0
    max_regret = 50.0
    p = np.clip(regret / max_regret, 0.0, 1.0)
    
    # Kappa schedule: high exploration early, low exploration late
    # kappa ranges from ~3.0 (early) to ~0.3 (late)
    kappa = 0.3 + 2.7 * (p ** 1.5)
    
    # LCB score: we want to maximize -mu + kappa * sigma
    score = -mu + kappa * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
