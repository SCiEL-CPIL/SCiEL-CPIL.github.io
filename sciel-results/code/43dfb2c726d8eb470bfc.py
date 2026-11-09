import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 285.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure sigma is safe
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Use Lower Confidence Bound (LCB) for exploitation/exploration balance
    # LCB = mu - beta * sigma (minimization: lower is better)
    # But we want higher score to be better, so we negate: -LCB = -mu + beta * sigma
    # This is essentially UCB for minimization
    
    # Adaptive beta: increase exploration when f_best is large (early), decrease when small (late)
    beta = 0.3 + 0.7 * np.exp(-f_best / 0.5)
    
    lcb_neg = -mu + beta * sigma_safe  # This is higher when uncertain (exploration)
    
    # Blend EI and LCB: EI is more principled for BO, LCB adds robustness
    # Weight EI more heavily as we get closer to the optimum
    w_ei = 0.7
    w_lcb = 0.3
    
    score = w_ei * ei + w_lcb * lcb_neg
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
