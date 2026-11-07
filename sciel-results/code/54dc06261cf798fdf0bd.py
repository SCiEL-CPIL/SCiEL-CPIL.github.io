"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _optimize_edges_for_values(values, m=35, deadline=None):
    """Find 35 edges that minimize max(distinct_sums, distinct_products)."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    best_raw = float('inf')
    best_edges = None
    
    # Greedy: start with first m edges, then try swaps
    current_edges = all_edges[:m]
    current_raw = _raw_value(values, current_edges)
    best_raw = current_raw
    best_edges = list(current_edges)
    
    # Try to improve by swapping edges
    edge_set = set(current_edges)
    remaining = all_edges[m:]
    
    for swap_idx in range(len(remaining)):
        if deadline is not None and time.time() > deadline:
            break
        new_edge = remaining[swap_idx]
        for remove_idx in range(m):
            old_edge = current_edges[remove_idx]
            if old_edge == new_edge:
                continue
            # Try swap
            trial_edges = list(current_edges)
            trial_edges[remove_idx] = new_edge
            trial_raw = _raw_value(values, trial_edges)
            if trial_raw < best_raw:
                best_raw = trial_raw
                best_edges = trial_edges
                current_edges = trial_edges
                break
    
    return best_edges, best_raw

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10 sec safety margin
    
    best_raw = float('inf')
    best_result = None
    
    # Generate candidate value sets
    value_candidates = []
    
    # 1. Small consecutive integers
    for offset in range(1, 50):
        vals = list(range(offset, offset + 14))
        if max(vals) <= 10000:
            value_candidates.append(vals)
    
    # 2. Arithmetic progressions with small steps
    for step in [2, 3, 4, 5]:
        for offset in range(1, 50):
            vals = [offset + i * step for i in range(14)]
            if max(vals) <= 10000:
                value_candidates.append(vals)
    
    # 3. Some random sets
    random.seed(42)
    for _ in range(20):
        vals = random.sample(range(1, 10001), 14)
        value_candidates.append(sorted(vals))
    
    # Evaluate each candidate
    for vals in value_candidates:
        if time.time() > deadline:
            break
        
        edges, raw = _optimize_edges_for_values(vals, 35, deadline)
        
        if raw < best_raw:
            best_raw = raw
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    # Fallback if no valid result found
    if best_result is None:
        vals = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        best_result = {'values': vals, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
