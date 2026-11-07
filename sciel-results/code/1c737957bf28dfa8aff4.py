# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    NLL-based Lower Confidence Bound (LCB) acquisition function.
    
    Strategy:
    Uses the Lower Confidence Bound (LCB) derived from the Negative Log-Likelihood (NLL)
    of the GP. This scales the exploration term by the local posterior variance in a 
    non-linear, heteroscedastic manner. Specifically, we use the form:
    
    LCB(x) = mu(x) - beta * sigma(x)
    
    where beta is a hyperparameter that controls the exploration-exploitation trade-off.
    Higher values of beta encourage more exploration. We adapt beta based on the 
    relationship between f_best and the scale of the problem to balance exploration 
    and exploitation dynamically.
    
    Rationale:
    In multimodal landscapes like Branin, standard EI can get trapped in local minima. 
    The LCB approach provides a principled way to balance exploration and exploitation 
    by directly minimizing a lower bound on the function value. This is particularly 
    effective when the GP model has high uncertainty in certain regions, as it will 
    naturally explore those areas.
    
    Implementation:
    1. Compute the LCB score as mu - beta * sigma
    2. Adapt beta based on f_best: when f_best is large (early in optimization), 
       use higher beta for more exploration; when f_best is small (late), use 
       lower beta for more exploitation
    3. Ensure all scores are finite
    """
    t_start = time.time()
    deadline = 0.1  # 100ms safety margin
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: balance exploration and exploitation
    # When f_best is large, we're early in optimization -> explore more (higher beta)
    # When f_best is small, we're converging -> exploit more (lower beta)
    # Branin's minimum is ~0.398, so f_best ranges from ~300 down to ~0.4
    # Use a smooth transition: beta = base_beta * (1 + log10(max(f_best, 1.0)))
    # This gives higher beta when f_best is large
    
    # Normalize f_best to get a sense of progress
    # f_best starts around 300 and decreases to ~0.4
    # Use log scale for better dynamic range
    f_best_norm = np.log10(max(f_best, 0.1))  # log10(300) ~ 2.5, log10(0.4) ~ -0.4
    
    # Adaptive beta: 
    # When f_best_norm is high (early), beta is higher (more exploration)
    # When f_best_norm is low (late), beta is lower (more exploitation)
    # Target: beta ~ 2-3 in early stages, beta ~ 0.5-1 in late stages
    beta = 1.5 + 0.8 * f_best_norm  # Adjusted to give reasonable range
    
    # Clamp beta to reasonable range
    beta = np.clip(beta, 0.3, 4.0)
    
    # Compute LCB scores: lower is better, so we negate for "higher is better" scoring
    # LCB = mu - beta * sigma
    # We want to maximize this (find the most promising point), so:
    scores = -(mu - beta * sigma)
    # Equivalent to: scores = -mu + beta * sigma
    
    # Add a small exploitation boost for points already better than f_best
    # This ensures we don't ignore a point that is clearly better
    improvement = np.maximum(0, f_best - mu)
    # Scale improvement by a small factor relative to the problem scale
    scale = max(f_best, 1.0)
    scores += 0.05 * improvement / scale
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If time is running out, return current
    if time.time() > t_start + deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
