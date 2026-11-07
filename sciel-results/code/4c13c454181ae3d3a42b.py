"""Search-based: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

DEADLINE = 85.0  # safety margin before 90s kill

def _raw_score(values, edges):
    """Compute max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    return list(itertools.combinations(range(n), 2))

def _greedy_edges(values, target_m):
    """Greedy edge selection: pick edges that minimize growth in distinct sums and products."""
    n = len(values)
    all_e = _all_edges(n)
    used = set()
    sums = set()
    prods = set()
    edges = []
    for _ in range(target_m):
        best_edge = None
        best_score = float('inf')
        for (u, v) in all_e:
            if (u, v) in used:
                continue
            a, b = values[u], values[v]
            s = a + b
            p = a * b
            new_s = 1 if s not in sums else 0
            new_p = 1 if p not in prods else 0
            score = max(len(sums) + new_s, len(prods) + new_p)
            if score < best_score:
                best_score = score
                best_edge = (u, v)
        if best_edge is None:
            break
        u, v = best_edge
        edges.append((u, v))
        used.add((u, v))
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return edges

def _local_search_edges(values, edges, target_m, max_iters=200):
    """Local search: try swapping one edge for another to reduce score."""
    n = len(values)
    all_e = _all_edges(n)
    current_score = _raw_score(values, edges)
    current_edge_set = set(edges)
    
    for _ in range(max_iters):
        improved = False
        # Try removing each edge and adding a better one
        for i, (u, v) in enumerate(edges):
            # Remove this edge
            temp_edges = edges[:i] + edges[i+1:]
            temp_edge_set = set(temp_edges)
            
            # Find best edge to add
            best_add = None
            best_new_score = current_score  # must improve
            
            for (a, b) in all_e:
                if (a, b) in temp_edge_set:
                    continue
                # Check if adding this edge improves score
                new_edges = temp_edges + [(a, b)]
                new_score = _raw_score(values, new_edges)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_add = (a, b)
            
            if best_add is not None:
                edges = temp_edges + [best_add]
                current_score = best_new_score
                improved = True
                break
        
        if not improved:
            break
    
    return edges, current_score

def _evaluate(values):
    """Evaluate a value set with greedy + local search edge selection."""
    edges = _greedy_edges(values, 35)
    if len(edges) < 35:
        return float('inf'), values, edges
    
    # Local search to improve
    edges, score = _local_search_edges(values, edges, 35)
    return score, values, edges

def _try_value_set(values):
    """Try a specific value set."""
    if len(set(values)) != 14:
        return None
    if any(not isinstance(v, int) or v < 1 or v > 10000 for v in values):
        return None
    score, vals, edges = _evaluate(values)
    return score, vals, edges

def _generate_structured_values():
    """Generate various structured value sets likely to have collisions."""
    candidates = []
    
    # Arithmetic progressions
    for a in range(1, 30):
        for d in range(1, 15):
            vals = [a + i * d for i in range(14)]
            if max(vals) <= 10000:
                candidates.append(vals)
    
    # Geometric progressions
    for a in range(1, 20):
        for r in range(2, 5):
            vals = []
            for i in range(14):
                v = a * (r ** i)
                if v > 10000:
                    break
                vals.append(v)
            if len(vals) == 14:
                candidates.append(vals)
    
    # Powers of 2
    vals = [2**i for i in range(14)]
    if max(vals) <= 10000:
        candidates.append(vals)
    
    # Multiples with small quotients
    for g in range(2, 20):
        for a in range(1, 10):
            for d in range(1, 5):
                qs = [a + i * d for i in range(14)]
                vals = [g * q for q in qs]
                if max(vals) <= 10000 and len(set(vals)) == 14:
                    candidates.append(vals)
    
    # Products of two small sequences
    for base in range(2, 10):
        vals = []
        for i in range(14):
            v = base * (i + 1)
            vals.append(v)
        if max(vals) <= 10000:
            candidates.append(vals)
    
    # Squares
    vals = [i*i for i in range(1, 15)]
    if max(vals) <= 10000:
        candidates.append(vals)
    
    # Cubes
    vals = [i**3 for i in range(1, 15)]
    if max(vals) <= 10000:
        candidates.append(vals)
    
    # Mixed: some small, some large
    for _ in range(50):
        small = random.sample(range(1, 20), 7)
        large = random.sample(range(500, 10001), 7)
        vals = small + large
        if len(set(vals)) == 14:
            candidates.append(vals)
    
    return candidates

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Strategy 1: Try structured value sets
    candidates = _generate_structured_values()
    for vals in candidates:
        if time.time() > DEADLINE:
            break
        result = _try_value_set(vals)
        if result:
            s, v, e = result
            if s < best_score:
                best_score = s
                best_result = (v, e)
    
    # Strategy 2: Random search with local improvement
    if time.time() < DEADLINE - 5:
        random.seed(42)
        for _ in range(300):
            if time.time() > DEADLINE - 5:
                break
            # Generate random distinct values
            vals = random.sample(range(1, 10001), 14)
            result = _try_value_set(vals)
            if result:
                s, v, e = result
                if s < best_score:
                    best_score = s
                    best_result = (v, e)
    
    # Strategy 3: Try to improve best result with more local search
    if best_result is not None and time.time() < DEADLINE - 2:
        vals, edges = best_result
        edges, score = _local_search_edges(vals, edges, 35, max_iters=500)
        if score < best_score:
            best_score = score
            best_result = (vals, edges)
    
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(14), 2))[:35]
        return {'values': vals, 'edges': [list(e) for e in edges]}
    
    vals, edges = best_result
    # Ensure proper types
    vals = [int(v) for v in vals]
    edges = [list(e) for e in edges]
    return {'values': vals, 'edges': edges}

# EVOLVE-BLOCK-END
