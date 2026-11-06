"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track instance and cache the learned model
_current_instance_id = None
_learned_alpha = None
_learned_beta = None
_instance_start_time = None
TIME_LIMIT = 115.0  # Safety margin from 120s limit

def _get_instance_id(weights, values):
    """Create a simple hashable ID for the instance to detect changes."""
    # Using the first few items and length as a proxy for ID
    # This is fast and sufficient for detecting if the problem instance changed
    if len(weights) > 3:
        return (len(weights), weights[0], weights[1], weights[2], values[0], values[1], values[2])
    return (len(weights), weights, values)

def _fit_linear_model(weights, values):
    """
    Fit a linear model value = alpha * weight + beta to the items.
    This captures the global trend of value vs weight.
    """
    # Convert to numpy arrays for fast linear algebra
    w = np.array(weights, dtype=np.float64)
    v = np.array(values, dtype=np.float64)
    
    # Solve least squares: [w, 1] * [alpha, beta]^T = v
    # Using np.linalg.lstsq for robustness
    A = np.column_stack((w, np.ones_like(w)))
    try:
        result, residuals, rank, s = np.linalg.lstsq(A, v, rcond=None)
        alpha, beta = result
    except:
        # Fallback to simple ratio if linear fit fails
        alpha = np.mean(v / w) if np.all(w > 0) else 0
        beta = 0
        
    return float(alpha), float(beta)

def _greedy_pack_with_model(remaining_capacity, remaining_weights, remaining_values, alpha, beta):
    """
    Simulate the greedy packing using the learned linear model.
    Returns the index of the first item picked in this simulated greedy sequence.
    """
    # We need to simulate the greedy choice sequence to see which item is picked first
    # that maximizes the *predicted* value based on the model, while respecting capacity.
    
    # Actually, a simpler and faster heuristic:
    # The "best" item to pick next is the one that, when added, leaves the most "efficient"
    # remaining capacity for the model.
    # However, the most direct interpretation of "global surrogate" is:
    # Score each item i by its "contribution to the linear fit" or simply its actual value,
    # but prioritize items that are "typical" of the high-value region.
    
    # Let's refine: 
    # The greedy algorithm using the *model* would pick items in order of (alpha*w_i + beta) / w_i = alpha + beta/w_i.
    # This is equivalent to picking items with the smallest weight (if beta > 0) or largest (if beta < 0)?
    # No, the slope is constant. The intercept dominates the density if beta is large.
    
    # A better surrogate approach:
    # 1. Calculate the "model value" for each item: v_model_i = alpha * w_i + beta
    # 2. Sort items by v_model_i / w_i (model density)
    # 3. Greedily pick the first feasible item in this sorted order.
    # 4. Return the index of that first item.
    
    # Optimization: We don't need to simulate the *whole* pack, just find the first item
    # that would be picked by the model-based greedy strategy.
    
    if len(remaining_weights) == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    
    # Model density: (alpha * w + beta) / w = alpha + beta / w
    # If beta is positive, smaller weights have higher model density.
    # If beta is negative, larger weights have higher model density.
    
    # To be safe and robust, let's just use the actual values but ranked by a composite score
    # that reflects the global fit. 
    # Actually, the most effective "global" heuristic often found in literature for knapsack
    # is to use the "correlation" between weight and value.
    
    # Let's stick to the simplest effective global model:
    # Sort by value/weight (standard greedy) but break ties or adjust based on the global slope?
    # No, let's use the *predicted* value from the linear fit as the sorting key,
    # but normalized by weight to get density.
    
    # Key insight: If the items follow a linear trend, the "density" is roughly constant.
    # Deviations from the trend (residuals) indicate "good" or "bad" items.
    # We should pick items with high *residuals* (actual value > predicted value).
    
    v_predicted = alpha * w + beta
    residuals = v - v_predicted
    
    # We want high actual value, but also "surprise" (high residual).
    # Let's score by: actual_value + residual * scaling_factor
    # Or simply sort by residuals? No, that ignores the base value.
    # Sort by (actual_value / weight) is the baseline.
    
    # New Strategy: Sort by "Adjusted Density" = (value + residual) / weight = 2*value/weight - (alpha*w+beta)/w
    # This effectively doubles the weight of the actual density.
    
    # Let's try a different structural change: **Look-ahead with Model**.
    # Pick the item that maximizes the total value of the *first k* items packed by the model.
    
    # For simplicity and speed, let's use the residual as a tie-breaker or primary sort key
    # if the densities are similar.
    
    # Let's calculate the standard density
    densities = v / w
    
    # Let's calculate the residual score
    res_score = residuals / w  # Residual density
    
    # Combine them: 0.5 * density + 0.5 * (density + res_score) = density + 0.5 * res_score
    # This prioritizes items that are denser than the global trend.
    
    combined_score = densities + 0.5 * res_score
    
    # Find the index of the max combined score
    # But we must ensure feasibility (w <= remaining_capacity)
    # The input arrays are already "remaining (feasible)"? 
    # "index into the remaining (feasible) arrays" -> implies all items in the array are feasible?
    # "remaining_weights" usually means items not yet picked, but are they all feasible?
    # The contract says "remaining (feasible) arrays". This suggests we only receive items that fit.
    # If so, we can just pick the max score.
    
    # However, to be safe, let's check capacity.
    # If the input is truly only feasible items, we don't need to filter.
    # But if it's all remaining items, we must filter.
    # Given "remaining (feasible) arrays", I will assume they are all feasible.
    
    best_idx = int(np.argmax(combined_score))
    return best_idx

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _current_instance_id, _learned_alpha, _learned_beta, _instance_start_time
    
    # Check time limit
    if _instance_start_time is None:
        _instance_start_time = time.time()
    elif time.time() - _instance_start_time > TIME_LIMIT:
        return -1
        
    # Check if instance changed
    instance_id = _get_instance_id(remaining_weights, remaining_values)
    
    if instance_id != _current_instance_id:
        _current_instance_id = instance_id
        _instance_start_time = time.time()
        
        # Fit the model on the full set of items (which are currently remaining)
        # Note: This is called on the *remaining* items. As items are removed,
        # the set changes. Should we refit every step?
        # Refitting every step is O(N^2) per instance (N steps * O(N) fit).
        # For N=100, that's 10,000 operations. Very fast.
        # But the "global" model should ideally be on the *original* set.
        # We don't have access to the original set, only remaining.
        # Fitting on remaining items adapts to the current state.
        
        _learned_alpha, _learned_beta = _fit_linear_model(remaining_weights, remaining_values)
    else:
        # Instance is the same, but items have been removed.
        # We could refit, but it might be expensive or noisy.
        # Let's refit every 10 steps or if the change is significant?
        # For now, let's refit every step to keep the model fresh, 
        # as N is small (100).
        if len(remaining_weights) > 1:
             _learned_alpha, _learned_beta = _fit_linear_model(remaining_weights, remaining_values)
        else:
             # If only 1 item left, linear fit is unstable, just pick it if feasible
             return 0 if len(remaining_weights) == 1 else -1

    if len(remaining_weights) == 0:
        return -1
        
    if len(remaining_weights) == 1:
        # Only one item left, check if it fits
        if remaining_weights[0] <= remaining_capacity:
            return 0
        else:
            return -1

    return _greedy_pack_with_model(remaining_capacity, remaining_weights, remaining_values, _learned_alpha, _learned_beta)
# EVOLVE-BLOCK-END
