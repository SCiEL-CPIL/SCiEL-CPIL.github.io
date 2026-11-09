# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a dynamic blend of Probability of Improvement (PI)
    and Lower Confidence Bound (LCB).
    
    Strategy:
    - Early in optimization (high uncertainty, low f_best relative to range), prioritize 
      exploration using LCB (mu - beta * sigma) to find promising regions quickly.
    - Later in optimization, prioritize exploitation using PI (P(f(x) < f_best)) to 
      refine the best candidate.
    - The transition is based on the ratio of the current best value to the initial 
      scale, or simply by the magnitude of sigma relative to f_best. 
    - To be robust without knowing the exact iteration, we use a heuristic based on 
      the normalized uncertainty.
    
    However, a simpler and more effective structural change is to use a **Thompson Sampling-like**
    approach or **PI with cost-awareness**.
    
    Let's stick to the PI vs LCB blend but make it adaptive based on the "confidence" of the GP.
    If sigma is high everywhere, we are in exploration mode. If sigma is low, we are in exploitation mode.
    
    Actually, the insight from the cell findings is that PI is better early.
    Let's use a soft switch based on the average sigma relative to the range of mu.
    
    1. Compute PI: P(f(x) < f_best) = norm.cdf((f_best - mu) / sigma)
    2. Compute LCB score: -(mu - beta * sigma) = beta * sigma - mu (higher is better)
    3. Normalize both to [0, 1] range roughly to combine them.
    4. Blend weight w_explore: 
       - If mean(sigma) is high relative to std(mu), we are uncertain -> high w_explore.
       - If mean(sigma) is low, we are certain -> low w_explore.
    
    Cost adjustment: Divide final score by cost.
    """
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        mu = np.where(np.isfinite(mu), mu, f_best)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Probability of Improvement (PI)
    # For minimization, we want P(f(x) < f_best)
    z_pi = (f_best - mu) / sigma_safe
    z_pi = np.clip(z_pi, -10, 10)
    pi = norm.cdf(z_pi)
    
    # 2. Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - beta * sigma.
    # So the "score" to maximize is beta * sigma - mu.
    # Let's use beta = 2.0 to encourage exploration.
    beta = 2.0
    lcb_score = beta * sigma_safe - mu
    
    # Normalize LCB score to be comparable to PI (which is in [0,1])
    # LCB score can be negative or positive. 
    # Let's map LCB score to a probability-like scale or just normalize by its range.
    # A simple normalization: (lcb_score - min) / (max - min) might be unstable.
    # Instead, let's use a sigmoid-like transformation or just scale it.
    # Since PI is [0,1], let's scale LCB so that typical values fall in [0,1] range.
    # If f_best is around 0-10, and sigma is around 1-5, LCB is roughly -10 to 10.
    # Let's use a softplus or similar to keep it positive and bounded-ish.
    # Actually, let's just normalize LCB by the current f_best magnitude to make it scale-invariant.
    # But f_best can be 0.
    
    # Alternative: Use a weighted sum where the weight adapts.
    # Let's estimate the "exploration need" based on the ratio of max sigma to f_best.
    # If f_best is large, we need more improvement. If sigma is large, we have uncertainty.
    
    # Let's try a different structural approach: **Expected Improvement with a Gaussian Process Kernel Trick**
    # No, let's stick to the PI/LCB blend but make the blend dynamic based on the **variance of mu**.
    # If mu varies a lot, the GP is confident in its predictions across the space -> Exploit (PI).
    # If mu is flat (low variance) but sigma is high, the GP is uncertain -> Explore (LCB).
    
    mu_var = np.var(mu)
    sigma_mean = np.mean(sigma_safe)
    
    # Heuristic:
    # If sigma_mean is high relative to the spread of mu, we are in the "flat prior" phase.
    # Ratio = sigma_mean / (std(mu) + 1e-6)
    ratio = sigma_mean / (np.std(mu) + 1e-6)
    
    # If ratio is high (> 1), we prioritize LCB. If low (< 0.5), we prioritize PI.
    # Map ratio to weight w_lcb in [0.2, 0.8]
    # Use a sigmoid centered at 1.0
    w_lcb = 1.0 / (1.0 + np.exp(-2.0 * (ratio - 1.0)))
    # Clamp weight to avoid extremes
    w_lcb = np.clip(w_lcb, 0.2, 0.8)
    w_pi = 1.0 - w_lcb
    
    # Normalize LCB score to be on a similar scale as PI (0 to 1)
    # We can use a simple scaling: lcb_score / (f_best + sigma_mean + 1e-6)
    # This makes it roughly dimensionless.
    scale_factor = (np.abs(f_best) + sigma_mean + 1e-6)
    lcb_norm = lcb_score / scale_factor
    
    # Clip lcb_norm to [0, 1] to make it a "score" comparable to PI
    # Note: lcb_norm can be negative. We want higher to be better.
    # If lcb_norm is negative, it's bad. If positive, it's good.
    # Let's shift it so that 0 is the baseline.
    # Or just use raw lcb_score but scaled down.
    
    # Let's try: Score = w_pi * PI + w_lcb * (LCB_score / scale_factor)
    # But PI is [0,1]. LCB_score/scale_factor is roughly [-2, 2].
    # Let's clip LCB part to [0, 1] as well for a true convex combination.
    lcb_clipped = np.clip(lcb_norm, 0.0, 1.0)
    
    scores = w_pi * pi + w_lcb * lcb_clipped
    
    # Cost adjustment
    scores = scores / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
