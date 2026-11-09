"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _optimize_edges(values, max_time=30):
    """Given values, find 35 edges minimizing max(distinct_sums, distinct_products)."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_data = []
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((i, j, s, p))
    
    # Start with all edges, then remove worst ones
    # Greedy: start with all 91 edges, remove 56 to get 35
    # Or start random and swap
    
    best_edges = None
    best_score = float('inf')
    
    # Try multiple random starts
    start_time = time.time()
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Random initial set of 35 edges
        edges = random.sample(all_edges, 35)
        edges_set = set(edges)
        
        current_score = _score(values, [list(e) for e in edges])
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            # Try swapping one edge in with one edge out
            random.shuffle(all_edges)
            for new_edge in all_edges:
                if new_edge in edges_set:
                    continue
                for old_edge in list(edges_set):
                    # Try swap
                    new_edges_list = [e for e in edges_set if e != old_edge]
                    new_edges_list.append(new_edge)
                    new_score = _score(values, [list(e) for e in new_edges_list])
                    if new_score < current_score:
                        edges_set.remove(old_edge)
                        edges_set.add(new_edge)
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(e) for e in edges_set]
    
    return best_edges, best_score

def _generate_structured_values():
    """Generate candidate value sets with good collision properties."""
    candidates = []
    
    # Arithmetic progression
    for start in range(1, 20):
        for step in range(1, 10):
            vals = [start + i * step for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14:
                candidates.append(vals)
    
    # Small values (1-50)
    for _ in range(100):
        vals = random.sample(range(1, 51), 14)
        candidates.append(vals)
    
    # Values with multiplicative structure
    for _ in range(100):
        # Pick values that are products of small primes
        base_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24]
        vals = random.sample(base_vals, 14)
        candidates.append(vals)
    
    # Random small values
    for _ in range(200):
        vals = random.sample(range(1, 101), 14)
        candidates.append(vals)
    
    return candidates

def construct():
    deadline = time.time() + 80  # 80 second budget, 10s safety margin
    
    best_result = None
    best_value = float('inf')
    
    # Generate candidate value sets
    value_sets = _generate_structured_values()
    random.shuffle(value_sets)
    
    for vals in value_sets:
        if time.time() > deadline:
            break
        
        # Ensure valid
        if len(set(vals)) != 14 or any(v < 1 or v > 10000 for v in vals):
            continue
        
        # Optimize edges for these values
        edges, score = _optimize_edges(vals, max_time=15)
        
        if edges is None:
            continue
        
        if score < best_value:
            best_value = score
            best_result = {'values': list(vals), 'edges': edges}
        
        if time.time() > deadline:
            break
    
    # Fallback
    if best_result is None:
        vals = list(range(1, 15))
        all_edges = list(combinations(range(14), 2))
        edges = [list(e) for e in all_edges[:35]]
        best_result = {'values': vals, 'edges': edges}
        best_value = _score(vals, edges)
    
    return best_result
# EVOLVE-BLOCK-END
