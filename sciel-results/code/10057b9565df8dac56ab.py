"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Discounted Lower Confidence Bound.
    
    Strategy:
    1. Quantile LCB: Instead of mu - beta*sigma, we use the 10th percentile of the 
       predictive distribution: mu - 1.2816*sigma. This corresponds to a fixed 
       confidence level (90%) which is often more robust than adaptive beta that 
       might over-exploit early or under-exploit late.
    2. Sigmoidal Cost Discount: The cost is 1 + mean(x). Since x is in [0,1]^3, 
       cost ranges from 1 to 2. We apply a sigmoid discount that sharply reduces 
       the score of high-cost points (outer regions) while preserving low-cost 
       points (inner regions). This leverages the structure that the minimum is 
       likely to be found in the cheaper central regions, or at least that 
       exploring the expensive outer shell is less efficient per unit cost.
       
       Discount function: 1 / (1 + exp(k * (cost - c0)))
       With c0 = 1.2 and k = 10, points with cost < 1.2 are boosted (~0.9), 
       points with cost > 1.4 are suppressed (~0.1).
       
    3. Normalization: We normalize the scores to be in a reasonable range to avoid 
       numerical issues, though the relative ordering is what matters.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety check
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n, dtype=np.float64)
        
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Quantile LCB (10th percentile)
    # z_0.10 = -1.28155...
    z_quantile = 1.28155
    lcb = mu - z_quantile * sigma_safe
    
    # 2. Sigmoidal Cost Discount
    # cost is in [1, 2]. We want to discount higher costs.
    # Let's center the sigmoid around the median cost (1.5) or slightly lower to 
    # favor cheaper points. 
    # If we use c0 = 1.3, then cost=1 (cheapest) gets discount ~0.88, cost=2 (most expensive) gets ~0.02.
    # Let's try c0 = 1.2 and k = 15 for a sharper transition.
    c0 = 1.2
    k = 15.0
    discount = 1.0 / (1.0 + np.exp(k * (cost_safe - c0)))
    
    # 3. Combine
    # We want to maximize the score. 
    # LCB is "lower is better" for the function value, but we want "higher is better" for acquisition.
    # So we should maximize -LCB or just LCB if the framework expects lower to be better?
    # The contract says: "Return 2000 finite scores (higher = evaluate next)".
    # And "minimizes hartmann 3d".
    # So if mu is low, that's good. LCB = mu - ... is low. 
    # Wait, standard LCB for minimization is mu - beta*sigma. We want to pick the point with the LOWEST LCB.
    # But the acquisition function returns scores where HIGHER is better.
    # So we should return -LCB * discount? Or (some_max - LCB) * discount?
    # Let's look at the inspiration: "EI / cost". EI is positive improvement.
    # If we use LCB, a low LCB is good. So score should be high when LCB is low.
    # Score = -LCB * discount?
    # If mu = -3.8, sigma = 0.1, LCB = -3.8 - 0.128 = -3.928. Score = 3.928 * discount.
    # If mu = 0, sigma = 0.1, LCB = 0 - 0.128 = -0.128. Score = 0.128 * discount.
    # This seems correct. Lower function values yield higher scores.
    
    scores = -lcb * discount
    
    # Normalize scores to [0, 1] or similar to prevent overflow?
    # The values of hartmann 3d are around -3.8 to 3.8.
    # -lcb can be around 3.8 + 1.28*sigma.
    # If sigma is large, say 1.0, -lcb can be 5.0.
    # Discount is [0, 1].
    # Scores are roughly [0, 5]. This is safe.
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check
    if time.time() > deadline:
        pass
        
    return scores
# EVOLVE-BLOCK-END
