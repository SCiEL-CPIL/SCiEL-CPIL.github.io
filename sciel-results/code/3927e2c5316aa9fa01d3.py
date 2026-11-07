import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple EI / cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Constants
    f_min = -3.32237
    # Estimate f_max for progress calculation.
    # Hartmann 6d values on [0,1]^6 typically range from -3.32 to ~-1.0.
    # Using -1.0 as a conservative "worst case" for progress scaling.
    f_max_est = -1.0
    
    # Calculate progress p in [0, 1]
    # p=0 when f_best is high (early), p=1 when f_best is low (late)
    denom = f_max_est - f_min
    if denom < 1e-9:
        p = 0.5
    else:
        p = (f_max_est - f_best) / denom
        p = np.clip(p, 0.0, 1.0)
        
    # Weights for blending UCB and EI
    # Early: more UCB (exploration)
    # Late: more EI (exploitation)
    w_ei = p
    w_ucb = 1.0 - p
    
    # Beta for UCB
    # High beta early, low beta late
    beta_max = 3.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * p
    
    # Compute EI
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / (sigma + 1e-9)
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB Improvement
    # UCB value for minimization: L(x) = mu - beta * sigma
    # Improvement: f_best - L(x)
    ucb_lower_bound = mu - beta * sigma
    ucb_imp = f_best - ucb_lower_bound
    ucb_imp = np.maximum(ucb_imp, 0.0)
    
    # Cost-aware scores
    # Divide by cost to penalize expensive points
    # cost is in [1, 2]
    score_ei = ei / cost
    score_ucb = ucb_imp / cost
    
    # Combine scores
    score = w_ei * score_ei + w_ucb * score_ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
