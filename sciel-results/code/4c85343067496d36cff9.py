import numpy as np
import time

_START = None
_HARD_LIMIT = 145.0  # seconds before the 150s kill; leave safety margin

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _fast_density_index(weights, values, feasible):
    """O(n) fallback: pick max value/weight among feasible."""
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    fd = np.where(feasible, density, -np.inf)
    if not np.isfinite(fd).any():
        return 0
    return int(np.argmax(fd))

def _simulated_annealing_core(cap, weights, values, time_budget=0.5):
    """
    Uses Simulated Annealing to find a high-quality subset of items 
    that fits within the capacity. This explores the solution space 
    more globally than simple greedy.
    
    Returns:
    selected_mask: boolean array indicating which items are in the 'core'
    core_value: total value of the core
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    
    # Initial solution: Density Greedy
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    
    order = np.argsort(-density, kind='stable')
    
    current_selected = np.zeros(n, dtype=bool)
    current_value = 0.0
    current_weight = 0.0
    
    cap = float(cap)
    
    # Build initial greedy solution
    for idx in order:
        idx = int(idx)
        if weights[idx] <= cap - current_weight:
            current_selected[idx] = True
            current_value += values[idx]
            current_weight += weights[idx]
            if current_weight >= cap:
                break
    
    best_selected = current_selected.copy()
    best_value = current_value
    
    # SA Parameters
    # Temperature schedule: Start high, decay exponentially
    # We don't know the scale of values, so normalize or use a heuristic start
    max_val = np.max(values) if n > 0 else 1.0
    if max_val == 0:
        return best_selected, best_value
        
    T_start = max_val * 0.5
    T_end = max_val * 0.01
    # Number of iterations depends on time budget, but we cap it to avoid overhead
    # n=100, so 1000-5000 iterations is reasonable
    max_iter = 2000
    alpha = (T_end / T_start) ** (1.0 / max_iter) if max_iter > 0 else 0.5
    
    T = T_start
    start_time = time.time()
    
    rng = np.random.default_rng()
    
    for i in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if T < T_end:
            break
            
        # Random move: pick a random item to toggle
        idx = rng.integers(0, n)
        
        if current_selected[idx]:
            # Try to remove
            new_weight = current_weight - weights[idx]
            new_value = current_value - values[idx]
            delta = -values[idx]
            # Always accept removal if it allows adding something else later? 
            # For SA on 0/1 knapsack, we usually swap or toggle.
            # If we remove, we gain capacity.
            # Let's just toggle.
            if np.random.rand() < 1.0: # Always accept removal? No, SA logic.
                # Actually, standard SA: accept if delta > 0 or with prob exp(delta/T)
                # Here delta is negative (loss in value).
                if np.random.rand() < np.exp(delta / T):
                    current_selected[idx] = False
                    current_weight = new_weight
                    current_value = new_value
        else:
            # Try to add
            if weights[idx] > cap - current_weight:
                # Infeasible, skip or swap? 
                # Simple SA: skip infeasible moves.
                continue
            new_weight = current_weight + weights[idx]
            new_value = current_value + values[idx]
            delta = values[idx]
            
            # Always accept improvement
            if delta > 0:
                current_selected[idx] = True
                current_weight = new_weight
                current_value = new_value
            else:
                # Accept with probability
                if np.random.rand() < np.exp(delta / T):
                    current_selected[idx] = True
                    current_weight = new_weight
                    current_value = new_value
                    
        if current_value > best_value:
            best_value = current_value
            best_selected = current_selected.copy()
            
        T *= alpha
        
    return best_selected, best_value

def _greedy_fill(remaining_cap, remaining_weights, remaining_values, excluded_mask):
    """
    Given a capacity and a set of items NOT in the core, greedily fill
    with best density items that fit.
    Returns:
    first_index: index of the first item added (global index) or -1
    total_value: value added
    """
    n = len(remaining_weights)
    if n == 0 or remaining_cap <= 0:
        return -1, 0.0
        
    # Filter out excluded items
    available = ~excluded_mask
    if not available.any():
        return -1, 0.0
        
    avail_indices = np.where(available)[0]
    avail_weights = remaining_weights[avail_indices]
    avail_values = remaining_values[avail_indices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(avail_weights > 0, avail_values / avail_weights, 0.0)
        
    order = np.argsort(-density, kind='stable')
    
    total_val = 0.0
    first_idx = -1
    cap = float(remaining_cap)
    
    for idx in order:
        global_idx = int(avail_indices[idx])
        w = float(avail_weights[idx])
        v = float(avail_values[idx])
        
        if w <= cap:
            cap -= w
            total_val += v
            if first_idx == -1:
                first_idx = global_idx
            if cap <= 0:
                break
                
    return first_idx, total_val

def _hybrid_select(cap, weights, values):
    """
    Combines SA core selection with Greedy Fill.
    Returns the index of the best first item to pick.
    """
    n = len(weights)
    if n == 0 or cap <= 0:
        return -1
        
    feasible = weights <= cap
    if not feasible.any():
        return -1
        
    # Time check
    elapsed = _now() - _START
    if elapsed > _HARD_LIMIT - 1.0:
        return _fast_density_index(weights, values, feasible)
        
    # 1. Find Core using SA
    # Allocate time based on remaining budget
    time_left = _HARD_LIMIT - elapsed
    sa_time = min(0.5, time_left * 0.2) # Spend up to 0.5s on SA
    
    core_mask, core_value = _simulated_annealing_core(cap, weights, values, sa_time)
    
    core_weight = np.sum(weights[core_mask])
    remaining_cap = cap - core_weight
    
    # 2. Fill remaining capacity with best density items from non-core
    first_fill_idx, fill_value = _greedy_fill(remaining_cap, weights, values, core_mask)
    
    # Total value of this strategy
    total_val = core_value + fill_value
    
    # Determine the "first item" to return.
    # The evaluator calls select_next_item repeatedly. 
    # We need to return an index that is part of the optimal solution found.
    # However, the evaluator doesn't know about our internal "core" vs "fill".
    # It just takes the index we return, removes it, and calls us again.
    
    # Strategy:
    # If the SA core is empty, we should probably just return the best density item (standard greedy).
    # If the SA core is not empty, we should return the "best" item from the core?
    # Or the item from the fill?
    
    # The issue: The evaluator removes the item we return.
    # If we return an item from the "core", the next call will have a slightly different state.
    # Our SA solution was for the *current* state.
    # If we pick an item from the core, the remaining items in the core are still good.
    # If we pick an item from the fill, the core remains intact.
    
    # Which is better?
    # Usually, high density items are good. 
    # Let's compare the "quality" of the first core item vs the first fill item.
    
    # Actually, a simpler and more robust approach for the "next item" interface:
    # The SA finds a good *set*. We should pick the item from that set that, 
    # if removed, leaves the most "regret" or is the most "essential".
    # Alternatively, just pick the highest density item from the union of Core and Fill?
    # No, that defeats the purpose of SA.
    
    # Let's look at the items in the Core.
    core_indices = np.where(core_mask)[0]
    
    if len(core_indices) == 0:
        # SA failed or found nothing, fallback to density greedy
        return _fast_density_index(weights, values, feasible)
        
    # Among the core items, which one should we pick?
    # If we pick one, the rest of the core is still a valid subset for the next step?
    # Not necessarily, because capacity changes.
    
    # Heuristic: Pick the item in the Core with the highest density.
    # Why? Because it's part of a globally optimized set, and among them,
    # density is a good proxy for "efficiency".
    
    core_densities = np.zeros(len(core_indices))
    for i, idx in enumerate(core_indices):
        w = weights[idx]
        v = values[idx]
        core_densities[i] = v / w if w > 0 else 0
        
    best_core_pos = np.argmax(core_densities)
    candidate_core_idx = int(core_indices[best_core_pos])
    
    # Compare with the first fill item?
    # If fill_value is 0, there's no fill.
    # If fill exists, is it better to take the fill item first?
    # If we take the fill item, the core remains untouched.
    # If we take the core item, the core is broken.
    
    # Let's evaluate the "regret" of not taking the core item vs the fill item.
    # Actually, let's just pick the highest value density item from the *entire* 
    # recommended set (Core U Fill).
    
    # Wait, the Fill items are selected *after* the Core is fixed.
    # The set of recommended items is Core U Fill.
    
    recommended_indices = np.unique(np.concatenate([core_indices, [first_fill_idx] if first_fill_idx != -1 else []]))
    
    if len(recommended_indices) == 0:
        return _fast_density_index(weights, values, feasible)
        
    # Pick the best density from the recommended set
    rec_densities = []
    for idx in recommended_indices:
        w = weights[idx]
        v = values[idx]
        rec_densities.append(v / w if w > 0 else 0)
        
    best_rec_pos = np.argmax(rec_densities)
    final_candidate = int(recommended_indices[best_rec_pos])
    
    # Sanity check: is it feasible?
    if weights[final_candidate] <= cap:
        return final_candidate
    else:
        return _fast_density_index(weights, values, feasible)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity is None:
        return -1
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    if n == 0 or cap <= 0:
        return -1
    
    feasible = weights <= cap
    if not feasible.any():
        return -1
        
    elapsed = _now() - _START
    if elapsed > _HARD_LIMIT:
        return _fast_density_index(weights, values, feasible)
        
    # Use the hybrid strategy
    return _hybrid_select(cap, weights, values)
