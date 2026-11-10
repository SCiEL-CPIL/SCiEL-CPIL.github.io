import numpy as np
import time
import heapq

_GLOBAL_START = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _greedy_fill(capacity, weights, values, exclude_idx=None):
    """Greedy fill by density, excluding a specific index. Returns (total_value, list_of_indices)."""
    n = len(weights)
    if capacity <= 0:
        return 0.0, []
    
    candidates = []
    for i in range(n):
        w = weights[i]
        v = values[i]
        if exclude_idx is not None and i == exclude_idx:
            continue
        if w > 0 and w <= capacity:
            candidates.append((v / w, v, w, i))
        elif w == 0 and v > 0:
            candidates.append((float('inf'), v, w, i))
    
    if not candidates:
        return 0.0, []
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    total_val = 0.0
    selected = []
    rem_cap = capacity
    
    for density, v, w, idx in candidates:
        if w <= rem_cap:
            total_val += v
            selected.append(idx)
            rem_cap -= w
            if rem_cap <= 0:
                break
    
    return total_val, selected


def _solve_knapsack_dp(capacity, weights, values, max_time=5.0):
    """
    Exact DP for 0/1 knapsack with weight scaling.
    Returns (best_value, list_of_indices_included) or None if too expensive.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    # Filter to feasible items
    feasible_indices = []
    for i in range(n):
        w = weights[i]
        v = values[i]
        if w > 0 and w <= capacity:
            feasible_indices.append(i)
        elif w == 0 and v > 0:
            feasible_indices.append(i)
    
    if not feasible_indices:
        return 0.0, []
    
    fw = np.array([weights[i] for i in feasible_indices], dtype=np.float64)
    fv = np.array([values[i] for i in feasible_indices], dtype=np.float64)
    m = len(fw)
    
    # Scale weights to integers
    # Find the scale factor: we want total_weight * scale <= capacity * scale
    # Use a scale that gives us reasonable DP size
    total_w = float(np.sum(fw))
    if total_w <= 0:
        return 0.0, []
    
    # Target DP size: ~100000 states max for reasonable speed
    target_states = 100000
    # scale = target_states / total_w, so capacity*scale gives us the DP width
    scale = max(1.0, target_states / total_w)
    cap_scaled = int(round(capacity * scale))
    w_scaled = np.round(fw * scale).astype(np.int64)
    
    # Ensure no weight exceeds capacity
    w_scaled = np.minimum(w_scaled, cap_scaled)
    
    if cap_scaled > 200000 or m * cap_scaled > 5000000:
        return None  # Too expensive
    
    # DP: dp[c] = max value achievable with capacity c
    # Use 1D DP for space efficiency
    dp = np.zeros(cap_scaled + 1, dtype=np.float64)
    # Track which items are included for reconstruction
    # We'll use a 2D approach for reconstruction: dp[i][c] = max value using first i items with capacity c
    # But that's memory heavy. Instead, we'll do forward DP and track choices.
    
    # Actually, for reconstruction, let's use the standard approach:
    # dp[i][c] = max value using items 0..i-1 with capacity c
    # To save memory, we can use a 2D array but only if m*cap_scaled is reasonable
    if m * (cap_scaled + 1) <= 10000000:
        dp2d = np.zeros((m + 1, cap_scaled + 1), dtype=np.float64)
        for i in range(1, m + 1):
            wi = w_scaled[i - 1]
            vi = fv[i - 1]
            for c in range(cap_scaled + 1):
                dp2d[i, c] = dp2d[i - 1, c]
                if wi <= c:
                    candidate = dp2d[i - 1, c - wi] + vi
                    if candidate > dp2d[i, c]:
                        dp2d[i, c] = candidate
        
        best_val = dp2d[m, cap_scaled]
        
        # Reconstruct
        selected = []
        c = cap_scaled
        for i in range(m, 0, -1):
            if dp2d[i, c] != dp2d[i - 1, c]:
                selected.append(feasible_indices[i - 1])
                c -= w_scaled[i - 1]
        
        selected.reverse()
        return float(best_val), selected
    else:
        return None


def _solve_knapsack_bnb(capacity, weights, values, max_time=5.0):
    """
    Branch and bound for 0/1 knapsack.
    Returns (best_value, best_indices).
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    feasible_indices = []
    for i in range(n):
        if weights[i] > 0 and weights[i] <= capacity:
            feasible_indices.append(i)
        elif weights[i] == 0 and values[i] > 0:
            feasible_indices.append(i)
    
    if not feasible_indices:
        return 0.0, []
    
    items = []
    for i in feasible_indices:
        if weights[i] > 0:
            items.append((values[i] / weights[i], values[i], weights[i], i))
        else:
            items.append((float('inf'), values[i], weights[i], i))
    items.sort(key=lambda x: x[0], reverse=True)
    
    def fractional_bound(start_idx, cap):
        bound = 0.0
        rem_cap = cap
        for j in range(start_idx, len(items)):
            density, v, w, idx = items[j]
            if w <= rem_cap:
                bound += v
                rem_cap -= w
            else:
                if w > 0:
                    bound += v * (rem_cap / w)
                break
        return bound
    
    best_value = 0.0
    best_indices = []
    
    greedy_val, greedy_idx = _greedy_fill(capacity, weights, values)
    best_value = greedy_val
    best_indices = list(greedy_idx)
    
    start_time = time.time()
    
    heap = []
    initial_bound = fractional_bound(0, capacity)
    heapq.heappush(heap, (-initial_bound, 0.0, capacity, 0, ()))
    
    nodes_explored = 0
    max_nodes = 50000
    
    while heap and nodes_explored < max_nodes:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        neg_bound, cur_val, cap_left, next_idx, selected = heapq.heappop(heap)
        bound = -neg_bound
        nodes_explored += 1
        
        if bound <= best_value:
            continue
        
        if next_idx >= len(items):
            if cur_val > best_value:
                best_value = cur_val
                best_indices = list(selected)
            continue
        
        density, v, w, idx = items[next_idx]
        
        if w <= cap_left:
            new_val = cur_val + v
            new_cap = cap_left - w
            new_selected = selected + (idx,)
            new_bound = new_val + fractional_bound(next_idx + 1, new_cap)
            if new_bound > best_value:
                heapq.heappush(heap, (-new_bound, new_val, new_cap, next_idx + 1, new_selected))
        
        new_bound = cur_val + fractional_bound(next_idx + 1, cap_left)
        if new_bound > best_value:
            heapq.heappush(heap, (-new_bound, cur_val, cap_left, next_idx + 1, selected))
    
    return best_value, best_indices


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    start_time = time.time()
    
    # Filter to feasible items
    feasible_indices = []
    for i in range(n):
        w = remaining_weights[i]
        v = remaining_values[i]
        if w > 0 and w <= remaining_capacity:
            feasible_indices.append(i)
        elif w == 0 and v > 0:
            return i  # Free value, take it immediately
    
    if not feasible_indices:
        return -1
    
    f_weights = np.array([remaining_weights[i] for i in feasible_indices], dtype=np.float64)
    f_values = np.array([remaining_values[i] for i in feasible_indices], dtype=np.float64)
    f_indices = np.array(feasible_indices, dtype=np.int64)
    
    m = len(f_indices)
    
    # Greedy baseline
    greedy_val, greedy_sel = _greedy_fill(remaining_capacity, f_weights, f_values)
    best_total = greedy_val
    best_first = f_indices[greedy_sel[0]] if greedy_sel else f_indices[0]
    
    elapsed = time.time() - start_time
    remaining_time = _TIME_LIMIT - (time.time() - _GLOBAL_START)
    if remaining_time < 15.0:
        # Not enough time for more search
        return int(best_first)
    
    # Try exact DP first (guarantees optimal solution if it fits)
    if elapsed < remaining_time - 10.0:
        dp_result = _solve_knapsack_dp(remaining_capacity, f_weights, f_values, 
                                        max_time=min(15.0, remaining_time - 15.0))
        if dp_result is not None:
            dp_val, dp_sel = dp_result
            if dp_val > best_total:
                best_total = dp_val
                if dp_sel:
                    # Pick the highest density item from the optimal solution
                    sel_sorted = sorted(dp_sel, key=lambda i: f_values[i] / f_weights[i] if f_weights[i] > 0 else float('inf'), reverse=True)
                    best_first = f_indices[sel_sorted[0]]
                    return int(best_first)
    
    elapsed = time.time() - start_time
    remaining_time = _TIME_LIMIT - (time.time() - _GLOBAL_START)
    
    # If DP didn't work, try B&B
    if elapsed < remaining_time - 10.0:
        bnb_val, bnb_sel = _solve_knapsack_bnb(remaining_capacity, f_weights, f_values, 
                                               max_time=min(20.0, remaining_time - 15.0))
        if bnb_val > best_total:
            best_total = bnb_val
            if bnb_sel:
                sel_sorted = sorted(bnb_sel, key=lambda i: f_values[i] / f_weights[i] if f_weights[i] > 0 else float('inf'), reverse=True)
                best_first = f_indices[sel_sorted[0]]
    
    # Candidate-based approach: try each top candidate as first pick with greedy completion
    elapsed = time.time() - start_time
    remaining_time = _TIME_LIMIT - (time.time() - _GLOBAL_START)
    if elapsed < remaining_time - 10.0 and best_total <= greedy_val + 1e-9:
        densities = f_values / f_weights
        top_candidates = np.argsort(-densities)[:min(50, m)]
        
        for cand_local in top_candidates:
            if remaining_weights[f_indices[cand_local]] > remaining_capacity:
                continue
            
            rem_cap = remaining_capacity - f_weights[cand_local]
            val, _ = _greedy_fill(rem_cap, f_weights, f_values, exclude_idx=cand_local)
            total = f_values[cand_local] + val
            
            if total > best_total:
                best_total = total
                best_first = f_indices[cand_local]
    
    return int(best_first)
