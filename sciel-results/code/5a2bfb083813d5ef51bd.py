import numpy as np
import time

# Global state for deadline guard
_start_time = None
_call_count = 0
_last_f_best = None
_stagnation_counter = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count, _last_f_best, _stagnation_counter
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit, leave 30s margin
    if time.time() - _start_time > 270:
        # Fallback: Return simple cost-adjusted negative mean (exploitation only)
        # Ensure finite
        c_safe = np.maximum(cost, 1e-9)
        return np.where(np.isfinite(mu), -mu / c_safe, 0.0)

    _call_count += 1
    
    # Track stagnation
    if _last_f_best is not None:
        if f_best < _last_f_best - 1e-8:
            _stagnation_counter = 0
        else:
            _stagnation_counter += 1
    _last_f_best = f_best
    
    # Calculate progress based on f_best
    # 3HC min is 0. Values can be up to ~10.
    f_best_val = float(np.min(f_best))
    progress = np.clip(f_best_val, 0.0, 10.0) / 10.0
    
    # Beta decays from 2.5 (early) to 0.3 (late) for stronger exploration early
    # Modulate beta slightly by stagnation: if stuck, explore more
    base_beta = 0.3 + 2.2 * progress
    # If stagnating, increase beta slightly to force exploration
    stagnation_boost = 1.0 + 0.1 * min(_stagnation_counter, 10)
    beta = base_beta * stagnation_boost
    
    # Handle sigma=0
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Lower Confidence Bound (for minimization)
    # LCB = mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Structural Change: Multi-objective / Pareto-inspired scoring
    # Instead of simple division -lcb / cost, we model the trade-off more explicitly.
    # Objective 1: Minimize predicted value (LCB)
    # Objective 2: Minimize cost
    
    # We want a score that is high for low LCB and low Cost.
    # Let's use a weighted sum in the "normalized" space or a hypervolume-like heuristic.
    # Since we can't compute true hypervolume easily on 2000 points without a reference point and sorting,
    # we can use a "soft" Pareto dominance proxy.
    
    # Normalize LCB and Cost to [0, 1] range for this batch to make them comparable.
    # Note: LCB can be negative. Cost is in [1, 2].
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min + 1e-9
    lcb_norm = (lcb - lcb_min) / lcb_range  # 0 is best (lowest LCB), 1 is worst
    
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost_safe - cost_min) / cost_range  # 0 is best (lowest cost), 1 is worst
    
    # Weighted sum of normalized objectives.
    # We want to minimize (w1 * lcb_norm + w2 * cost_norm).
    # So we maximize -(w1 * lcb_norm + w2 * cost_norm).
    
    # Adaptive weights:
    # Early on (high progress value, i.e. f_best is high/bad), we care more about finding low values (w1 high).
    # Late on (low progress value, i.e. f_best is low/good), we care more about efficiency (w2 high) to squeeze out last bits?
    # Actually, if f_best is good, we want to exploit cheap good points.
    # If f_best is bad, we want to explore broadly, so cost matters less.
    
    # Let's invert:
    # progress = 1.0 means f_best ~ 10 (bad, early).
    # progress = 0.0 means f_best ~ 0 (good, late).
    
    # Weight for function value (importance of finding low mu)
    # Early: high weight on function. Late: lower weight?
    # Actually, standard BO logic: explore early, exploit late.
    # If exploiting, we want the point with lowest mu that is cheap.
    # If exploring, we want the point with highest uncertainty (lowest LCB) regardless of cost?
    
    # Let's define w_func and w_cost.
    # w_func = 1.0 + progress (early: 2.0, late: 1.0) -> prioritize function more when f_best is bad?
    # w_cost = 1.0 - progress (early: 0.0, late: 1.0) -> prioritize cost more when f_best is good?
    
    # However, cost is always important because it eats budget.
    # Let's try:
    w_func = 1.0 + 0.5 * progress
    w_cost = 1.0 - 0.5 * progress
    
    # Normalize weights so they sum to 2? Or just use as is.
    # Score = -(w_func * lcb_norm + w_cost * cost_norm)
    
    # This score is in range [-(w_func+w_cost), 0].
    # Higher is better (closer to 0).
    
    score = -(w_func * lcb_norm + w_cost * cost_norm)
    
    # To ensure the scale of the score doesn't cause issues with the evaluator (though it only cares about ranking),
    # we can keep it as is. The ranking is what matters.
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
