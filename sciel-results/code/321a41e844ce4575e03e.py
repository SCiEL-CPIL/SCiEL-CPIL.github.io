import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit
    elapsed = time.time() - _start_time
    if elapsed > 295:
        return np.zeros_like(mu)

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Strategy: Knowledge Gradient (KG) approximation with adaptive cost awareness
    # 
    # 1. Core Acquisition: Probability of Improvement (POI)
    #    POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    #    Unlike EI, POI is non-zero everywhere, providing a global exploration signal.
    #    In high dimensions, EI often collapses to 0 in unexplored regions, leading to stagnation.
    #    POI maintains a positive score for any point with non-negligible uncertainty or low mean.
    #
    # 2. Cost Awareness: Adaptive Scaling
    #    The cost of evaluation is cost(x) = 1 + mean(x_scaled). Range [1, 2].
    #    We want to prefer low-cost points, but not exclusively.
    #    A static exponential gate (like the current program) might be too aggressive or passive.
    #    
    #    Instead, we use a "Value of Information" (VOI) perspective.
    #    The benefit of evaluating x is the probability it improves f_best.
    #    The cost is proportional to cost(x).
    #    We define a dynamic beta that scales with the current uncertainty gap.
    #    
    #    Let gap = f_best - min_possible_estimate.
    #    We don't know min_possible_estimate exactly, but we can use a heuristic.
    #    However, a simpler robust approach is to normalize the cost penalty.
    #    
    #    Let's use a score: Score = POI * (1 / (1 + lambda * (cost - 1)))
    #    Here (cost - 1) is the extra cost over the minimum (range 0 to 1).
    #    lambda controls how strongly we penalize extra cost.
    #    
    #    To make it adaptive:
    #    If we are early on (high uncertainty, f_best is likely far from optimum), we should explore more,
    #    even at higher cost, to find the basin.
    #    If we are late (f_best is good), we should exploit cheap points.
    #    
    #    We can estimate "progress" by looking at the typical sigma or the spread of mu.
    #    Alternatively, just use a moderate fixed lambda that balances the 1-2 cost range.
    #    With cost in [1,2], (cost-1) in [0,1].
    #    If lambda = 1, a max cost point gets 1/2 weight of a min cost point.
    #    If lambda = 2, max cost gets 1/3 weight.
    #    
    #    Let's try lambda = 1.5. This is a significant discount for expensive points,
    #    but POI's global nature compensates by keeping scores positive.
    #
    #    Additionally, we can add a small "optimism" term to handle the case where sigma is very small
    #    but mu is still slightly above f_best (though POI handles this via CDF).
    #    Actually, POI is sufficient.
    #
    #    Let's refine: 
    #    z = (f_best - mu) / sigma
    #    poi = norm.cdf(z)
    #    
    #    Cost penalty:
    #    extra_cost = cost - 1.0  # Range [0, 1]
    #    penalty = 1.0 / (1.0 + 1.5 * extra_cost)
    #    
    #    score = poi * penalty
    
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Adaptive lambda: 
    # If the average sigma is large (early stage), we can afford higher cost to explore.
    # If average sigma is small (late stage), we should stick to cheap exploitation.
    # avg_sigma = np.mean(sigma_safe)
    # 
    # Let's use a simple heuristic: 
    # If f_best is still "high" (bad), explore more. 
    # We don't know the true minimum, but we know the theoretical min is -0.8.
    # Let's assume the target is around -0.8.
    # gap = f_best - (-0.8)
    # 
    # If gap is large, lambda should be small (less penalty).
    # If gap is small, lambda should be large (more penalty).
    #
    # lambda = 0.5 + 2.0 * (gap / max_gap) ??? 
    # 
    # Let's keep it simple but effective:
    # The current program uses exp(-k*cost). 
    # This new approach uses a rational discount.
    # 
    # Let's try a dynamic lambda based on the fraction of points that are "expensive".
    # No, let's just use a fixed moderate lambda first, as the structural change is POI vs EI.
    
    lambda_cost = 1.0
    extra_cost = np.maximum(cost - 1.0, 0.0)
    cost_discount = 1.0 / (1.0 + lambda_cost * extra_cost)
    
    score = poi * cost_discount
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
