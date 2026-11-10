import numpy as np
import time

_DEADLINE = 115.0
_START_TIME = None


def _get_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _time_left():
    return _DEADLINE - (time.time() - _get_start())


def _greedy_fill_value(weights, values, capacity, exclude_idx=None):
    """Compute total value from greedy density fill, optionally excluding one index."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    if len(w) == 0:
        return 0.0
    # Filter to feasible items
    feasible = w <= capacity
    if not np.any(feasible):
        return 0.0
    w = w[feasible]
    v = v[feasible]
    if len(w) == 0:
        return 0.0
    density = v / w
    order = np.argsort(-density)
    w = w[order]
    v = v[order]
    total_val = 0.0
    total_weight = 0.0
    for i in range(len(w)):
        if total_weight + w[i] <= capacity:
            total_weight += w[i]
            total_val += v[i]
        else:
            break
    return total_val


def _lookahead_evaluate(weights, values, capacity, top_k=15):
    """Evaluate top-k density candidates by greedy completion, return best index."""
    n = len(weights)
    if n == 0:
        return -1
    # Filter feasible
    feasible_mask = (weights > 0) & (weights <= capacity)
    if not np.any(feasible_mask):
        return -1
    w = weights[feasible_mask]
    v = values[feasible_mask]
    idx = np.where(feasible_mask)[0]
    if len(w) == 0:
        return -1
    density = v / w
    order = np.argsort(-density)
    w = w[order]
    v = v[order]
    idx = idx[order]
    
    k = min(top_k, len(w))
    best_val = -1.0
    best_idx = -1
    for i in range(k):
        # Evaluate: take item i, then greedy fill the rest
        val = v[i]
        rem_cap = capacity - w[i]
        # Greedy fill remaining
        for j in range(len(w)):
            if j == i:
                continue
            if rem_cap <= 0:
                break
            if w[j] <= rem_cap:
                rem_cap -= w[j]
                val += v[j]
            else:
                break
        if val > best_val:
            best_val = val
            best_idx = idx[i]
    return best_idx


def _local_search(weights, values, capacity, initial_sel, max_time=0.5):
    """1-swap and 2-for-1 local search to improve a solution."""
    n = len(weights)
    sel = list(initial_sel)
    sel_set = set(sel)
    sel_weight = sum(weights[i] for i in sel)
    sel_val = sum(values[i] for i in sel)
    
    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        # 1-swap: replace one selected with one unselected
        for out_idx in list(sel):
            if (time.time() - start) >= max_time:
                break
            out_w = weights[out_idx]
            out_v = values[out_idx]
            for in_idx in range(n):
                if in_idx in sel_set:
                    continue
                in_w = weights[in_idx]
                in_v = values[in_idx]
                if sel_weight - out_w + in_w <= capacity:
                    new_val = sel_val - out_v + in_v
                    if new_val > sel_val:
                        sel.remove(out_idx)
                        sel.append(in_idx)
                        sel_set.discard(out_idx)
                        sel_set.add(in_idx)
                        sel_weight = sel_weight - out_w + in_w
                        sel_val = new_val
                        improved = True
                        break
    
    # 2-for-1: remove two selected, add one unselected
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        sel_list = list(sel)
        for a in range(len(sel_list)):
            if (time.time() - start) >= max_time:
                break
            for b in range(a + 1, len(sel_list)):
                out1 = sel_list[a]
                out2 = sel_list[b]
                out_w = weights[out1] + weights[out2]
                out_v = values[out1] + values[out2]
                for in_idx in range(n):
                    if in_idx in sel_set:
                        continue
                    in_w = weights[in_idx]
                    in_v = values[in_idx]
                    if sel_weight - out_w + in_w <= capacity:
                        new_val = sel_val - out_v + in_v
                        if new_val > sel_val:
                            sel.remove(out1)
                            sel.remove(out2)
                            sel.append(in_idx)
                            sel_set.discard(out1)
                            sel_set.discard(out2)
                            sel_set.add(in_idx)
                            sel_weight = sel_weight - out_w + in_w
                            sel_val = new_val
                            improved = True
                            break
    return sel, sel_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Handle zero-weight positive-value items
    for i in range(n):
        if remaining_weights[i] == 0 and remaining_values[i] > 0:
            return i
    
    tl = _time_left()
    if tl <= 0.1:
        return -1
    
    # Prepare feasible items
    feasible_mask = (np.array(remaining_weights) > 0) & (np.array(remaining_weights) <= remaining_capacity)
    if not np.any(feasible_mask):
        return -1
    
    w = np.array(remaining_weights)[feasible_mask]
    v = np.array(remaining_values)[feasible_mask]
    idx = np.where(feasible_mask)[0]
    m = len(w)
    
    if m == 1:
        return idx[0]
    
    # Determine budget for this call
    # We expect up to 100 calls per instance, 8 instances = 800 calls
    # Total budget ~115s, so ~0.14s per call
    # But early calls have more items, later calls have fewer
    # Use a fraction of remaining time
    per_call_budget = min(0.12, max(0.02, tl / 100.0))
    
    if m <= 20:
        # Exact B&B for small problems
        items = list(zip(v, w, idx))
        items.sort(key=lambda x: x[0] / x[1], reverse=True)
        m2 = len(items)
        
        best_val = 0
        best_sel = []
        start = time.time()
        
        def dfs(d, curr_val, curr_weight, sel):
            nonlocal best_val, best_sel
            if (time.time() - start) > per_call_budget:
                return
            if d == m2:
                if curr_val > best_val:
                    best_val = curr_val
                    best_sel = list(sel)
                return
            # Upper bound with fractional relaxation
            ub = curr_val
            rem_cap = remaining_capacity - curr_weight
            for j in range(d, m2):
                vi, wi, _ = items[j]
                if wi <= rem_cap:
                    ub += vi
                    rem_cap -= wi
                else:
                    ub += vi * (rem_cap / wi)
                    break
            if ub <= best_val:
                return
            vi, wi, orig = items[d]
            if curr_weight + wi <= remaining_capacity:
                sel.append(orig)
                dfs(d + 1, curr_val + vi, curr_weight + wi, sel)
                sel.pop()
            dfs(d + 1, curr_val, curr_weight, sel)
        
        dfs(0, 0, 0, [])
        if best_sel:
            return best_sel[0]
        return -1
    else:
        # For larger problems: lookahead + local search
        # First, get a greedy solution
        density = v / w
        order = np.argsort(-density)
        w_sorted = w[order]
        v_sorted = v[order]
        idx_sorted = idx[order]
        
        greedy_sel = []
        greedy_weight = 0.0
        greedy_val = 0.0
        for i in range(m):
            if greedy_weight + w_sorted[i] <= remaining_capacity:
                greedy_weight += w_sorted[i]
                greedy_val += v_sorted[i]
                greedy_sel.append(idx_sorted[i])
        
        if not greedy_sel:
            return -1
        
        # Lookahead: try top-k candidates as first pick
        k = min(12, m)
        best_overall_val = greedy_val
        best_overall_sel = list(greedy_sel)
        
        for i in range(k):
            # Build solution starting with item i
            sel_i = [idx_sorted[i]]
            weight_i = w_sorted[i]
            val_i = v_sorted[i]
            # Greedy fill rest
            for j in range(m):
                if j == i:
                    continue
                if weight_i + w_sorted[j] <= remaining_capacity:
                    weight_i += w_sorted[j]
                    val_i += v_sorted[j]
                    sel_i.append(idx_sorted[j])
                elif (time.time() - start) > per_call_budget * 0.5 if 'start' in dir() else False:
                    break
            if val_i > best_overall_val:
                best_overall_val = val_i
                best_overall_sel = sel_i
        
        # Local search on the best solution found
        if (time.time() - _get_start()) < _DEADLINE - 0.5:
            ls_budget = min(per_call_budget * 0.4, max(0.01, _time_left() * 0.01))
            best_overall_sel, best_overall_val = _local_search(
                np.array(remaining_weights), np.array(remaining_values),
                remaining_capacity, best_overall_sel, max_time=ls_budget
            )
        
        if best_overall_sel:
            return best_overall_sel[0]
        return -1
