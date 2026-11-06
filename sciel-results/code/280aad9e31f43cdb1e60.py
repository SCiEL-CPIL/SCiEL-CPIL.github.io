import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 115.0

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

def _greedy_complete(w, v, cap, policy='density'):
    """
    Simulate greedy completion. Returns (total_value, list_of_indices).
    w, v are sorted by density desc.
    """
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0, []
    
    if policy == 'density':
        # Already sorted
        order = np.arange(n)
    elif policy == 'value':
        order = np.argsort(-v)
    else:
        order = np.arange(n)
        
    total_val = 0.0
    current_cap = cap
    selected = []
    
    for idx in order:
        if w[idx] <= current_cap:
            total_val += v[idx]
            current_cap -= w[idx]
            selected.append(idx)
            
    return total_val, selected

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
       cap = float(remaining_capacity)
       if cap <= 0:
           return -1
       
       w = np.asarray(remaining_weights, dtype=np.float64)
       v = np.asarray(remaining_values, dtype=np.float64)
       n = len(w)
       if n == 0:
           return -1
       
       # Filter feasible
       mask = w <= cap
       if not np.any(mask):
           return -1
       
       feasible_indices = np.where(mask)[0]
       w_f = w[mask]
       v_f = v[mask]
       n_f = len(w_f)
       
       # Sort by density descending
       densities = v_f / np.maximum(w_f, 1e-12)
       order = np.argsort(-densities)
       w_s = w_f[order]
       v_s = v_f[order]
       
       # Map sorted index -> original index in input
       # feasible_indices[order] gives the original index in `remaining_weights`
       orig_ids = feasible_indices[order]
       
       # Budget check
       if _remaining_budget() < 0.1:
           # Emergency: pick best density
           return int(orig_ids[0])
       
       # Lagrangian Relaxation / Shadow Price Approach
       # We want to find lambda such that sum(w_i * I(v_i - lambda*w_i > 0)) ~= cap
       # This is equivalent to finding the break-even density.
       
       # 1. Determine the critical density lambda*
       # Sort items by density descending.
       # Fill capacity greedily. The item that doesn't fit (or partially fits) determines lambda.
       current_cap = cap
       lambda_star = 0.0
       for i in range(n_f):
           if w_s[i] <= current_cap:
               current_cap -= w_s[i]
           else:
               # This item is the marginal one
               # lambda = v_i / w_i is the density.
               # Actually, the condition v_i - lambda*w_i > 0 is equivalent to v_i/w_i > lambda.
               # So lambda should be the density of the marginal item.
               lambda_star = densities[order[i]] if i < len(densities) else 0
               # Wait, `densities` was computed on w_f, v_f before sorting.
               # `order` maps sorted index to index in w_f.
               # So density of sorted item i is densities[order[i]]
               break
       
       # If all items fit, lambda_star is 0 (or very small)
       if current_cap > 0:
           lambda_star = 0.0
           
       # 2. Select items where v_i - lambda_star * w_i > 0
       # This is the set of items that are "profitable" at the shadow price.
       # In a continuous relaxation, we'd take all these.
       # In 0/1, we take those that fit.
       
       # However, this might select too many or too few.
       # The Lagrangian dual provides a lower bound on the optimal value.
       # The set of items selected by the Lagrangian relaxation (those with positive reduced cost)
       # is a strong candidate for the optimal solution.
       
       # Let's identify the "core" items: those with v_i - lambda_star * w_i > 0.
       reduced_costs = v_s - lambda_star * w_s
       core_mask = reduced_costs > 0
       
       # If core is empty, fallback to best density
       if not np.any(core_mask):
           return int(orig_ids[0])
           
       core_indices = np.where(core_mask)[0]
       
       # Now, among these core items, we need to pick the best NEXT item.
       # The heuristic suggests picking the item that maximizes the Lagrangian objective?
       # Or simply picking from the core set is sufficient?
       
       # Let's refine:
       # The optimal solution is likely contained within the core set.
       # If we pick an item from the core set, we are likely on the right track.
       # Which one?
       # A common heuristic is to pick the item with the highest reduced cost (v - lambda*w).
       # Or the highest density.
       
       # Let's try: Pick the item in the core set with the highest reduced cost.
       # If ties, higher value.
       
       # But wait, if lambda_star is the density of the marginal item,
       # then for items with density > lambda_star, reduced cost > 0.
       # For items with density == lambda_star, reduced cost = 0.
       # So we are picking from items with density > lambda_star.
       # This is essentially the greedy set, minus the marginal item.
       
       # This might be too restrictive if the marginal item is actually part of the optimal solution
       # (which happens if it fits perfectly or if swapping is beneficial).
       
       # Let's include the marginal item in consideration.
       # So consider items where density >= lambda_star - epsilon.
       # Actually, let's just consider the top K items by density, where K is determined by capacity.
       
       # Alternative Strategy:
       # Use the Lagrangian to guide a local search or B&B on a reduced problem.
       # But B&B on 100 items is slow.
       
       # Let's stick to the directive: "select the item that maximizes the Lagrangian relaxation objective".
       # Objective: v_i - lambda * w_i.
       # If we pick the max of this, we are picking the "most profitable" item at the current shadow price.
       
       # Let's compute the score for all feasible items.
       # score_i = v_i - lambda_star * w_i
       
       # But we must ensure we don't pick an item that makes it impossible to pick other high-value items.
       # The Lagrangian approach assumes we can take a fraction.
       
       # Let's try a hybrid:
       # 1. Identify core set (reduced cost > 0).
       # 2. If core set size is small (< 20), do B&B on core set + neighbors.
       # 3. Else, pick best reduced cost.
       
       # Given the budget, let's try B&B on the core set if it's small enough.
       
       # Determine core set size
       n_core = len(core_indices)
       
       # If core is too large, just pick best reduced cost
       if n_core > 30:
           # Pick max reduced cost
           scores = reduced_costs[core_indices]
           best_pos = int(np.argmax(scores))
           chosen_sorted_idx = core_indices[best_pos]
           return int(orig_ids[chosen_sorted_idx])
       
       # If core is small, do B&B on core + marginal items
       # Include items with density close to lambda_star
       # Let's just take the top 30 items by density, regardless of core status, to be safe.
       # But sort them by density.
       
       # Actually, the core set IS the top items by density (since lambda is the marginal density).
       # So core_indices are the indices in the sorted array [0, ..., k-1] where k is the first one that doesn't fit?
       # Not exactly. Lambda is the density of the item that *would* be picked next if we had infinite capacity?
       # No, Lambda is determined by the greedy fill.
       
       # Let's just run B&B on the first min(n_f, 40) items by density.
       # This is a heuristic reduction.
       
       max_bnb_items = min(n_f, 40)
       w_bnb = w_s[:max_bnb_items]
       v_bnb = v_s[:max_bnb_items]
       ids_bnb = orig_ids[:max_bnb_items]
       
       best_val = 0
       best_solution = []
       
       start_time = time.monotonic()
       timeout = min(5.0, _remaining_budget() * 0.5) # Dynamic timeout
       
       # Precompute suffix sums for bounds? No, simple bound is fine for N=40.
       # Actually, for N=40, B&B can be slow if not pruned well.
       # Let's use a simple recursive B&B.
       
       def bb(idx, cur_cap, cur_val, path):
           nonlocal best_val, best_solution
           
           if time.monotonic() - start_time > timeout:
               return
               
           if idx == max_bnb_items:
               if cur_val > best_val:
                   best_val = cur_val
                   best_solution = path[:]
               return
               
           # Bound: fractional knapsack
           b_val = cur_val
           b_cap = cur_cap
           for j in range(idx, max_bnb_items):
               if w_bnb[j] <= b_cap:
                   b_val += v_bnb[j]
                   b_cap -= w_bnb[j]
               else:
                   b_val += v_bnb[j] * (b_cap / w_bnb[j])
                   break
                   
           if b_val <= best_val:
               return
               
           # Branch: Take
           if w_bnb[idx] <= cur_cap:
               path.append(idx)
               bb(idx + 1, cur_cap - w_bnb[idx], cur_val + v_bnb[idx], path)
               path.pop()
               
           # Branch: Skip
           bb(idx + 1, cur_cap, cur_val, path)
           
       bb(0, cap, 0.0, [])
       
       if best_solution:
           # Pick the item from best_solution with max value? Or just the first one?
           # Let's pick the one with max value to be consistent with previous logic.
           vals = v_bnb[best_solution]
           max_pos = int(np.argmax(vals))
           chosen_sorted_idx = best_solution[max_pos]
           return int(ids_bnb[chosen_sorted_idx])
       else:
           # Fallback
           return int(orig_ids[0])
