#    
#    Is there any risk with normalization?
#    If all EI are 0 (convergence), std is 0, eps handles it.
#    If all LCB are same, std is 0, eps handles it.
#    
#    This looks robust.
#    
#    One more thing: The directive mentions "time-varying exploration parameter".
#    Since I can't access t, I'm using a fixed beta.
#    Is there a way to estimate t?
#    If I use a global counter, I can.
#    Let's try the global counter with a reset heuristic.
#    Heuristic: If `f_best` is greater than 1.0 (or some threshold), assume it's a new run or early stage.
#    Actually, Beale function values can be large initially.
#    If I reset the counter when `f_best` is "high", I might reset too often.
#    If I don't reset, the counter grows for seed 1 and 2, making beta too large.
#    Large beta -> too much exploration -> bad for later steps.
#    
#    Given the risk, fixed beta is safer.
#    The "structural change" is the hybrid normalization.
#    
#    Let's refine the beta.
#    In many BO papers for 2D, beta=2-5 works.
#    Let's try beta=2.0.
#    
#    Also, consider the weights.
#    Maybe 0.6 EI and 0.4 LCB?
#    EI is generally more reliable for convergence.
#    Let's try 0.6 EI, 0.4 LCB.
#    
#    Final plan:
#    - Hybrid of normalized EI and normalized LCB score.
#    - Weights: 0.6 EI, 0.4 LCB.
#    - Beta: 2.0.
#    - Deadline guard.

import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > 300:
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower Confidence Bound (for minimization)
    # We want to minimize, so we look at the lower bound of the function value.
    # LCB = mu - beta * sigma
    # We want to maximize the score, so we invert the LCB: -LCB = -mu + beta * sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # Normalize EI
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + eps
    ei_norm = (ei - ei_mean) / ei_std
    
    # Normalize LCB score
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score) + eps
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # Combine with weights
    # Slightly favor EI for convergence
    score = 0.6 * ei_norm + 0.4 * lcb_norm
    
    return score
# EVOLVE-BLOCK-END
