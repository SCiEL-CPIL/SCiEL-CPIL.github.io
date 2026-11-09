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
    z = (f_best - mu) / (sigma + eps)
    
    # EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Determine exploration/exploitation balance
    sigma_med = np.median(sigma)
    improvement_scale = f_best - np.min(mu)  # max possible improvement
    if improvement_scale < eps:
        improvement_scale = eps
    
    # Beta for UCB: larger when uncertainty is high relative to improvement
    beta = 2.0 * np.sqrt(np.log(1 + sigma_med / (improvement_scale + eps)))
    beta = np.clip(beta, 0.5, 5.0)
    
    # UCB-based improvement (optimistic for minimization)
    ucb_imp = np.maximum(f_best - mu + beta * sigma, 0)
    
    # Blend: when sigma is large relative to improvement, trust UCB more
    w_ucb = sigma_med / (sigma_med + improvement_scale + eps)
    w_ucb = np.clip(w_ucb, 0.3, 0.7)
    
    score = (1 - w_ucb) * ei + w_ucb * ucb_imp
    
    # Cost-aware: divide by cost, with slight amplification
    gamma = 1.0 + 0.2 * w_ucb  # more cost-sensitive when exploring
    score = score / (cost ** gamma)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
