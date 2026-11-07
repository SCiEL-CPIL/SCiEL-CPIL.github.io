"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _count_distinct(vals, edges):
    """Count distinct sums and products."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = vals[u], vals[v]
        sums.add(a + b)
        prods.add(a * b)
    return len(sums), len(prods)

def _best_edges_for_values(values, m=35, deadline=None):
    """Find m edges minimizing max(distinct sums, distinct products)."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_data = []
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data.append((u, v, s, p))
    
    best_score = m + 1  # worst case
    best_edge_set = None
    
    # Greedy approach: try to pick edges that reuse existing sums/products
    # Start with a random subset, then try to improve
    random.seed(42)
    
    # Try multiple random starts
    for attempt in range(50):
        if deadline and time.time() > deadline - 5:
            break
        
        # Random initial subset
        current_edges = random.sample(all_edges, m)
        current_sums = set()
        current_prods = set()
        for u, v in current_edges:
            current_sums.add(values[u] + values[v])
            current_prods.add(values[u] * values[v])
        current_score = max(len(current_sums), len(current_prods))
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(current_edges)
        
        # Try to improve by swapping edges
        for swap_iter in range(200):
            if deadline and time.time() > deadline - 5:
                break
            if best_score <= 1:
                break
            
            # Try swapping one edge out and one in
            if not current_edges:
                break
            out_idx = random.randint(0, len(current_edges) - 1)
            out_edge = current_edges[out_idx]
            
            # Find a candidate to swap in
            candidates = [e for e in all_edges if e not in current_edges]
            if not candidates:
                break
            in_edge = random.choice(candidates)
            
            # Compute new score
            new_sums = set(current_sums)
            new_prods = set(current_prods)
            # Remove old edge's contribution (need to check if it's the only edge with that sum/prod)
            old_u, old_v = out_edge
            old_s = values[old_u] + values[old_v]
            old_p = values[old_u] * values[old_v]
            
            # Count how many edges have this sum/prod
            sum_count = 0
            prod_count = 0
            for u, v in current_edges:
                if u == old_u and v == old_v:
                    continue
                if values[u] + values[v] == old_s:
                    sum_count += 1
                if values[u] * values[v] == old_p:
                    prod_count += 1
            
            if sum_count == 0:
                new_sums.discard(old_s)
            if prod_count == 0:
                new_prods.discard(old_p)
            
            new_u, new_v = in_edge
            new_s = values[new_u] + values[new_v]
            new_p = values[new_u] * values[new_v]
            new_sums.add(new_s)
            new_prods.add(new_p)
            
            new_score = max(len(new_sums), len(new_prods))
            
            if new_score < current_score:
                # Accept the swap
                current_edges[out_idx] = in_edge
                current_sums = new_sums
                current_prods = new_prods
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edge_set = set(current_edges)
    
    if best_edge_set is None:
        best_edge_set = set(random.sample(all_edges, m))
    
    return list(best_edge_set), best_score

def _try_value_set(values, deadline=None):
    """Try a given value set and return best (score, values, edges)."""
    edges, score = _best_edges_for_values(values, 35, deadline)
    return score, values, edges

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    best_score = 999
    best_values = None
    best_edges = None
    
    # Strategy 1: Try small consecutive values with different edge selections
    for base in range(1, 20):
        if time.time() > deadline - 10:
            break
        values = list(range(base, base + 14))
        score, vals, edges = _try_value_set(values, deadline)
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
    
    # Strategy 2: Try values with shared factors (multiples of small numbers)
    for multiplier in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
        # Generate 14 distinct multiples
        multiples = []
        for k in range(1, 100):
            val = multiplier * k
            if 1 <= val <= 10000 and val not in multiples:
                multiples.append(val)
            if len(multiples) >= 14:
                break
        if len(multiples) == 14:
            values = multiples
            score, vals, edges = _try_value_set(values, deadline)
            if score < best_score:
                best_score = score
                best_values = vals
                best_edges = edges
    
    # Strategy 3: Random value sets
    random.seed(123)
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        values = random.sample(range(1, 10001), 14)
        score, vals, edges = _try_value_set(values, deadline)
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
    
    # Strategy 4: Try geometric-like sequences
    for start_val in [2, 3, 4]:
        if time.time() > deadline - 10:
            break
        values = []
        val = start_val
        while len(values) < 14 and val <= 10000:
            values.append(val)
            val *= 2
        if len(values) == 14:
            score, vals, edges = _try_value_set(values, deadline)
            if score < best_score:
                best_score = score
                best_values = vals
                best_edges = edges
    
    # Fallback
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges, best_score = _best_edges_for_values(best_values, 35, deadline)
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
