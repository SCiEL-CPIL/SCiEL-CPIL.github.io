import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard: we must return well before 330s wall-clock
    deadline = time.time() + 30.0  # be very conservative, this function should be fast
    
    # Handle shape
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)  # (100,)
    else:
        P = np.asarray(prizes).flatten()
        
    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1)
        
    n_constraints, n_items = W.shape
    
    # Per-constraint efficiency: for each constraint c and item i
    # eff[c, i] = P[i] / W[c, i]
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = P[None, :] / (W + 1e-12)
    eff = np.where(np.isfinite(eff), eff, 0.0)
    
    # Scarcity estimation: for each constraint, look at the distribution of efficiencies
    # Constraints where the top items are much more efficient than average are "scarcer"
    # because those few items dominate that resource's usage
    
    # Top-k efficiency per constraint (k = top 20%)
    k = max(1, n_items // 5)
    top_k_eff = np.zeros(n_constraints)
    mean_eff = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_k_eff[c] = np.partition(eff[c], -k)[-k].mean()
        mean_eff[c] = eff[c].mean()
    
    # Scarcity score: high top-k/mean ratio indicates a constraint is "tight"
    # because a few items are much better at using that resource
    with np.errstate(divide='ignore', invalid='ignore'):
        scarcity_ratio = top_k_eff / (mean_eff + 1e-12)
    scarcity_ratio = np.where(np.isfinite(scarcity_ratio), scarcity_ratio, 1.0)
    
    # Normalize scarcity weights to sum to 1
    scarcity = scarcity_ratio / (scarcity_ratio.sum() + 1e-12)
    
    # Marginal utility: weighted sum of per-constraint efficiencies
    marginal_util = (scarcity[:, None] * eff).sum(axis=0)
    
    # Base efficiency: total prize / total weight
    W_sum = W.sum(axis=0)
    base_eff = P / (W_sum + 1e-12)
    base_eff = np.where(np.isfinite(base_eff), base_eff, 0.0)
    
    # Soft-max bottleneck penalty: penalize items that are heavy in any single constraint
    # Using log-sum-exp for smoothness
    beta = 3.0
    W_max = np.max(W, axis=0)
    log_sum_exp = W_max + np.log(np.sum(np.exp((W - W_max) * beta), axis=0)) / beta
    soft_bottleneck = log_sum_exp
    
    # NEW: Constraint-specific scarcity premium
    # For each item, identify the most binding constraint (largest weight fraction)
    # and scale its efficiency by the inverse of remaining capacity fraction
    
    # Weight fraction: W[c, i] / W_sum[i]
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_frac = W / (W_sum[None, :] + 1e-12)
    weight_frac = np.where(np.isfinite(weight_frac), weight_frac, 0.0)
    
    # Most binding constraint per item (argmax of weight fraction)
    most_binding = np.argmax(weight_frac, axis=0)  # shape (n_items,)
    
    # For each item, get its most binding constraint's weight and capacity
    # Capacity is 25 for all constraints
    capacity = 25.0
    most_binding_weight = W[most_binding, np.arange(n_items)]
    most_binding_frac = weight_frac[most_binding, np.arange(n_items)]
    
    # Remaining capacity fraction: 1 - (most_binding_weight / capacity)
    # This tells us how much of the most limiting resource is left
    with np.errstate(divide='ignore', invalid='ignore'):
        remaining_frac = 1.0 - (most_binding_weight / capacity)
    remaining_frac = np.clip(remaining_frac, 0.01, 1.0)  # Avoid division by zero
    
    # Scarcity premium: 1 / remaining_frac
    # Higher when the most binding constraint is nearly full
    scarcity_premium = 1.0 / remaining_frac
    
    # Also consider the overall scarcity of that specific constraint
    # Use the scarcity ratio for the most binding constraint
    constraint_scarcity = scarcity_ratio[most_binding]
    
    # Combine: scale the item's efficiency by both the local scarcity premium
    # and the global scarcity of its most binding constraint
    item_specific_premium = scarcity_premium * constraint_scarcity
    
    # Normalize the premium to have mean 1 for stability
    item_specific_premium = item_specific_premium / (item_specific_premium.mean() + 1e-12)
    
    # Combine all components:
    # - marginal_util: scarcity-weighted efficiency
    # - base_eff: overall efficiency
    # - soft_bottleneck: penalty for heavy items
    # - item_specific_premium: new constraint-specific scarcity premium
    H = marginal_util * base_eff * item_specific_premium / (soft_bottleneck + 1e-12)
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    # Deadline guard
    if time.time() > deadline - 5.0:
        return H
    
    return H
