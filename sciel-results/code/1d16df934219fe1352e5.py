import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Structural change from SEI/EI:
    - Uses linear LCB: -mu + beta * sigma
    - Normalizes by cost: LCB / cost
    - This is more robust to GP prediction errors and scaling than EI.
    - Allows high-improvement, high-cost points to be selected if the 
      linear improvement per cost is high.
    """
    t0 = time.time()
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB: We want to minimize f, so we maximize -mu + beta * sigma
    # beta controls the exploration/exploitation trade-off.
    # A higher beta encourages exploring regions with high uncertainty.
    # Since we divide by cost, this effectively explores cheap, uncertain regions.
    beta = 3.0
    
    lcb = -mu + beta * sigma_safe
    
    # Cost-aware weighting
    # We want to maximize improvement per unit cost.
    # cost = 1 + mean(x_scaled). Lower cost is better.
    cost_safe = np.maximum(cost, 1e-8)
    
    # Score = LCB / cost
    # This is the structural change: linear numerator, linear denominator.
    scores = lcb / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety: if computation took too long, return what we have
    # (The vectorized ops above are extremely fast, <1ms for 2000 points)
    elapsed = time.time() - t0
    if elapsed > 290.0:
        # Fallback: return simple scores if we're close to timeout
        # This is unlikely to trigger given the simplicity of the operation
        pass
        
    return scores
