"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Hard Feasibility Gate.
    
    Strategy:
    1. Compute Expected Improvement (EI) for all candidates.
    2. Compute a "Gain per Cost" metric (EI / cost).
    3. Apply a hard gate based on cost percentiles:
       - Candidates with cost in the top 70% (cheapest 30%) are allowed to be selected.
       - Candidates with cost in the bottom 30% (most expensive 30%) are allowed to be selected ONLY if 
         their EI is significantly higher than the best EI in the cheap zone, OR if the cheap zone has 
         low uncertainty (converged). 
       - However, a simpler structural change observed in inspirations is to simply restrict 
         evaluation to the low-cost corner until uncertainty is exhausted. 
       
       Let's implement a strict "Low-Cost First" gate:
       - Calculate the 70th percentile of the cost array.
       - Any candidate with cost > percentile_70 gets a score of -inf (or very low).
       - Among the remaining candidates (cost <= percentile_70), we rank by EI / cost.
       
       This forces the search to concentrate evaluations in the low-cost region.
       As the GP model improves in the low-cost region, the EI there will drop.
       If the EI in the low-cost region drops below a threshold relative to the global best,
       we might need to open the gate. But for a structural change, let's stick to the 
       percentile gate first. To prevent total stagnation if the minimum is in a high-cost region,
       we can blend: 
       Score = (EI / cost) * Gate
       Where Gate is 1.0 if cost < P70, and a small epsilon (e.g., 0.1) otherwise.
       This preserves the possibility of finding a high-cost minimum but heavily penalizes it.
       
       Let's refine based on the "Hard Multiplicative Feasibility Gate" finding:
       "Replacing smooth cost division with a hard multiplicative feasibility gate (quantile-based) 
        forces the search to concentrate evaluations in the low-cost corner... Measured: 0.8783 -> 1.697."
        
       So, we use a hard gate.
       Gate = 1 if cost <= P70 else 0.
       Score = EI / cost if Gate == 1 else -inf.
       
       Wait, if the global minimum is in the high-cost region, this will fail.
       The Levy 2d function is multimodal. The minimizer is moved to a uniform point in the inner 80%.
       The cost is 1 + mean(x_scaled). x_scaled is in [0,1].
       The cost is determined by the position.
       If the minimum is at x=0.9, cost is high.
       If we never evaluate high-cost regions, we will never find it.
       
       However, the inspiration says this improved the score. Why?
       Perhaps because the GP is better at predicting in the low-cost region initially?
       Or perhaps the "inner 80%" constraint means the minimum is not in the extreme corners?
       Inner 80% of [-10, 10] is [-8, 8]. Scaled to [0,1], that's [0.1, 0.9].
       Cost = 1 + mean(x_scaled).
       If min is at 0.1, cost ~ 1.1. If min is at 0.9, cost ~ 1.9.
       The difference is significant.
       
       Let's try a softer version of the hard gate to avoid total failure if the min is expensive.
       Score = (EI / cost) * (1.0 if cost < P70 else 0.05)
       
       This is a structural change from the previous smooth 1/cost weighting.
    """
    # Deadline guard
    if time.time() - _start_time > 290:
        # Fallback to simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Compute Cost-Aware Score
    cost_safe = np.maximum(cost, 1e-9)
    base_score = ei / cost_safe
    
    # Apply Hard Feasibility Gate
    # Find the 70th percentile of costs
    # We want to focus on the cheapest 30% of candidates.
    # So we gate out the most expensive 70%? No, "low-cost corner".
    # If we keep the cheapest 30%, we use cost <= P30.
    # The inspiration mentioned "70th/30th percentile thresholds".
    # "Forces the search to concentrate evaluations in the low-cost corner... 70th/30th percentile thresholds".
    # This likely means:
    # Zone 1 (Exploit): Cost <= P30 (Cheapest 30%) -> Full weight
    # Zone 2 (Explore): Cost > P70 (Most expensive 30%) -> Low weight or specific condition
    # Zone 3 (Middle): P30 < Cost <= P70 -> Excluded?
    
    # Let's implement the exclusion of the middle to force a bimodal search:
    # Either go very cheap (exploit known low-cost areas) or go very expensive (explore unexplored high-cost areas).
    # Or, more likely, the "hard gate" just blocks high costs.
    
    # Let's try: Block the top 70% of costs (i.e., only allow the cheapest 30% to be selected).
    # This is the most aggressive "low-cost corner" strategy.
    p30 = np.percentile(cost, 30)
    
    # Mask: 1 if cost is in the cheapest 30%, 0 otherwise.
    # To allow some exploration, we can give a small epsilon to the rest.
    # But the finding says "hard multiplicative feasibility gate".
    # Let's use a hard 0 for the expensive ones, but to be safe against the minimum being in the expensive region,
    # we will use a very small epsilon (1e-6) instead of 0, so that if EI is huge, it might still pick it?
    # No, EI is bounded by f_best. If f_best is 10, EI is max 10.
    # 10 / 1.9 (max cost) ~ 5.
    # If we set score to 0 for expensive, we will never pick them.
    
    # Let's follow the finding's implication: "concentrate evaluations in the low-cost corner... until GP uncertainty there is exhausted".
    # This implies we should stop evaluating low-cost areas once they are well-modeled.
    # EI naturally drops as uncertainty drops.
    # So, EI/cost will drop.
    # If we hard-gate to low-cost, we will exhaust it. Then all scores are ~0.
    # Then we will pick the max of ~0s, which might be random or the least bad.
    # This is bad.
    
    # Better structural change: **Probability of Improvement (PI) with Cost Gate**.
    # PI is more exploratory than EI in some regimes? No, PI is less exploitative.
    # Let's stick to the gate idea but make it adaptive.
    
    # Adaptive Gate:
    # Calculate the mean EI in the low-cost zone (cheapest 30%).
    # If this mean EI is high, keep the gate strict.
    # If this mean EI is low, relax the gate to allow exploring other areas.
    
    low_cost_mask = cost <= p30
    if np.any(low_cost_mask):
        mean_ei_low = np.mean(ei[low_cost_mask])
    else:
        mean_ei_low = 0.0
        
    # Threshold for relaxing the gate
    # If the expected improvement in the cheap zone is less than 1% of f_best, we are "exhausted".
    # f_best is the current best value.
    threshold = 0.01 * abs(f_best) + 1e-6
    
    if mean_ei_low > threshold:
        # Exploit cheap zone
        # Score = EI / cost for cheap, -inf for expensive
        scores = np.full_like(base_score, -np.inf)
        scores[low_cost_mask] = base_score[low_cost_mask]
    else:
        # Explore everywhere, but still prefer cheap
        # Use standard EI/cost
        scores = base_score
        
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
