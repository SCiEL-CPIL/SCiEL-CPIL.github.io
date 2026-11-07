"""Search-based: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

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

def _greedy_edges_fast(values, target_m):
    """Greedy edge selection with O(1) set-based membership check."""
    n = len(values)
    all_e = _all_edges(n)
    sums = set()
    prods = set()
    edges = []
    edge_set = set()
    for _ in range(target_m):
        best_edge = None
        best_score = float('inf')
        for (u, v) in all_e:
            if (u, v) in edge_set:
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
        edge_set.add((u, v))
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return edges

def _local_search_edges(values, edges, max_iters=20):
    """Local search: try swapping edges to reduce distinct sums/products."""
    n = len(values)
    all_e = _all_edges(n)
    edge_set = set(edges)
    current_score = _raw_score(values, edges)
    
    for _ in range(max_iters):
        improved = False
        for idx, (u, v) in enumerate(edges):
            # Compute state without this edge
            sums = set()
            prods = set()
            for (a, b) in edges:
                if (a, b) == (u, v):
                    continue
                sums.add(values[a] + values[b])
                prods.add(values[a] * values[b])
            
            # Try adding each non-edge
            for (x, y) in all_e:
                if (x, y) in edge_set:
                    continue
                s = values[x] + values[y]
                p = values[x] * values[y]
                new_score = max(len(sums | {s}), len(prods | {p}))
                if new_score < current_score:
                    edges[idx] = (x, y)
                    edge_set.discard((u, v))
                    edge_set.add((x, y))
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return edges, current_score

def _evaluate(values):
    """Evaluate a value set with greedy edge selection plus local search polish."""
    edges = _greedy_edges_fast(values, 35)
    if len(edges) < 35:
        return float('inf'), values, edges
    edges, score = _local_search_edges(values, edges, max_iters=15)
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
    """Generate structured value sets likely to have many collisions."""
    candidates = []
    
    # Strategy A: Common factor g, quotients in AP
    for g in range(2, 50):
        for a in range(1, 15):
            for d in range(1, 8):
                qs = [a + i * d for i in range(14)]
                vals = [g * q for q in qs]
                if max(vals) <= 10000 and len(set(vals)) == 14:
                    candidates.append(vals)
    
    # Strategy B: Geometric-like progressions with common factor
    for g in range(2, 30):
        for base in range(2, 10):
            for ratio_num in range(3, 8):
                for ratio_den in range(1, ratio_num):
                    vals = []
                    current = g * base
                    for _ in range(14):
                        vals.append(current)
                        current = (current * ratio_num) // ratio_den
                    if max(vals) <= 10000 and len(set(vals)) == 14 and min(vals) >= 1:
                        candidates.append(vals)
    
    # Strategy C: Powers of 2 times small integers
    for g in range(1, 20):
        for offset in range(0, 5):
            vals = [g * (2**i + offset) for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14 and min(vals) >= 1:
                candidates.append(vals)
    
    # Strategy D: Multiples with structured quotients (Sidon-like)
    for g in range(2, 40):
        # Use quadratic residues or similar structured sets
        for start in range(1, 10):
            for step in range(1, 5):
                qs = [start + i * step + (i * i) // 4 for i in range(14)]
                vals = [g * q for q in qs]
                if max(vals) <= 10000 and len(set(vals)) == 14 and min(vals) >= 1:
                    candidates.append(vals)
    
    return candidates

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Phase 1: Try structured value sets
    structured = _generate_structured_values()
    random.shuffle(structured)
    
    for vals in structured:
        if time.time() > DEADLINE - 5:
            break
        result = _try_value_set(vals)
        if result:
            s, v, e = result
            if s < best_score:
                best_score = s
                best_result = (v, e)
    
    # Phase 2: Simulated annealing on value sets
    if time.time() < DEADLINE - 10:
        random.seed(42)
        
        # Start from best found or a random set
        if best_result:
            current_vals = list(best_result[0])
            current_score = best_score
        else:
            current_vals = random.sample(range(1, 10001), 14)
            current_score = float('inf')
            result = _try_value_set(current_vals)
            if result:
                current_score, current_vals, _ = result
        
        temp = 5.0
        cooling = 0.95
        
        while time.time() < DEADLINE - 10 and temp > 0.1:
            # Perturb current value set
            new_vals = list(current_vals)
            # Randomly modify 1-3 values
            num_mods = random.randint(1, 3)
            for _ in range(num_mods):
                idx = random.randint(0, 13)
                if random.random() < 0.5:
                    # Small perturbation
                    delta = random.randint(-20, 20)
                    new_vals[idx] = max(1, min(10000, new_vals[idx] + delta))
                else:
                    # Larger jump
                    new_vals[idx] = random.randint(1, 10000)
            
            # Ensure distinct
            if len(set(new_vals)) != 14:
                temp *= cooling
                continue
            
            result = _try_value_set(new_vals)
            if not result:
                temp *= cooling
                continue
            
            new_score, _, _ = result
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_vals = new_vals
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_result = (list(new_vals), result[2])
            
            temp *= cooling
    
    # Phase 3: Final local search refinement on best result
    if best_result and time.time() < DEADLINE - 3:
        vals, edges = best_result
        # Do additional local search iterations
        for _ in range(5):
            if time.time() > DEADLINE - 3:
                break
            edges, score = _local_search_edges(vals, edges, max_iters=10)
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
