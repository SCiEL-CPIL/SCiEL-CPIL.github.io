"""Model-and-optimize: parameterize 14 values and 35 edges; fit against max(distinct_sums, distinct_products)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    """Compute the raw score: max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    """Generate all C(n,2) possible edges."""
    return list(itertools.combinations(range(n), 2))

def _compute_edge_attrs(values, all_edges):
    """Precompute sum and product for each edge."""
    attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        attrs[(u, v)] = (a + b, a * b)
    return attrs

def _greedy_edge_selection(values, target_m, all_edges):
    """
    Greedy: pick edges that maximize reuse of existing sum/product values.
    """
    n = len(values)
    edges_chosen = []
    chosen_set = set()
    sum_counts = {}
    prod_counts = {}
    
    # Precompute all edge attributes
    edge_info = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        edge_info.append((u, v, s, p))
    
    for _ in range(target_m):
        best_edge = None
        best_key = None
        for (u, v, s, p) in edge_info:
            if (u, v) in chosen_set:
                continue
            s_count = sum_counts.get(s, 0)
            p_count = prod_counts.get(p, 0)
            key = (s_count + p_count, -s, -p)
            if best_key is None or key > best_key:
                best_key = key
                best_edge = (u, v, s, p)
        
        if best_edge is None:
            break
        
        u, v, s, p = best_edge
        edges_chosen.append((u, v))
        chosen_set.add((u, v))
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    return [list(e[:2]) for e in edges_chosen]

def _fast_local_search(values, edges, all_edges, deadline, max_iters=500):
    """
    Fast local search: edge swaps with adaptive swap count.
    Uses precomputed edge attributes for speed.
    """
    n = len(values)
    m = len(edges)
    
    # Precompute edge attributes once
    edge_attrs = _compute_edge_attrs(values, all_edges)
    
    current_edges = [list(e) for e in edges]
    edge_set = set(tuple(e) for e in current_edges)
    
    # Current score state
    sum_counts = {}
    prod_counts = {}
    for u, v in current_edges:
        s, p = edge_attrs[(u, v)]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    n_sums = len(sum_counts)
    n_prods = len(prod_counts)
    current_score = max(n_sums, n_prods)
    
    all_edge_list = list(all_edges)
    
    no_improve_count = 0
    swap_attempts = 15
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try edge swaps
        for i in range(m):
            if time.time() > deadline:
                break
            
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
            # Try random swaps
            for _ in range(swap_attempts):
                e2 = random.choice(all_edge_list)
                if e2 == e1 or e2 in edge_set:
                    continue
                
                s2, p2 = edge_attrs[e2]
                
                old_s_count = sum_counts[s1]
                old_p_count = prod_counts[p1]
                
                new_n_sums = n_sums
                new_n_prods = n_prods
                if old_s_count == 1:
                    new_n_sums -= 1
                if old_p_count == 1:
                    new_n_prods -= 1
                
                new_s_count = sum_counts.get(s2, 0)
                new_p_count = prod_counts.get(p2, 0)
                if new_s_count == 0:
                    new_n_sums += 1
                if new_p_count == 0:
                    new_n_prods += 1
                
                new_score = max(new_n_sums, new_n_prods)
                
                if new_score < current_score:
                    sum_counts[s1] = old_s_count - 1
                    if sum_counts[s1] == 0:
                        del sum_counts[s1]
                    prod_counts[p1] = old_p_count - 1
                    if prod_counts[p1] == 0:
                        del prod_counts[p1]
                    
                    sum_counts[s2] = new_s_count + 1
                    prod_counts[p2] = new_p_count + 1
                    
                    n_sums = new_n_sums
                    n_prods = new_n_prods
                    current_score = new_score
                    
                    current_edges[i] = list(e2)
                    edge_set.remove(e1)
                    edge_set.add(e2)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            # Reduce swap attempts when stuck
            if no_improve_count > 5:
                swap_attempts = max(5, swap_attempts - 3)
            if no_improve_count > 40:
                break
    
    return current_score, current_edges

def _optimize_values(values, edges, deadline, max_iters=200):
    """
    Local search on values: perturb individual values to reduce distinct sums/products.
    """
    n = len(values)
    m = len(edges)
    current_values = list(values)
    
    # Precompute edge index pairs
    edge_pairs = [(u, v) for u, v in edges]
    
    def _score(vals):
        sums = set()
        prods = set()
        for u, v in edge_pairs:
            a, b = vals[u], vals[v]
            sums.add(a + b)
            prods.add(a * b)
        return max(len(sums), len(prods))
    
    current_score = _score(current_values)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try perturbing each value
        for i in range(n):
            if time.time() > deadline:
                break
            
            old_val = current_values[i]
            # Try small perturbations
            for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20]:
                new_val = old_val + delta
                if new_val <= 0 or new_val > 10000:
                    continue
                if new_val in current_values:
                    continue
                
                current_values[i] = new_val
                new_score = _score(current_values)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    current_values[i] = old_val
            
            if improved:
                break
        
        if not improved and iteration > 20:
            break
    
    return current_score, current_values

def _make_value_set_1():
    """Powers of 2 and 3 for structured products."""
    candidates = []
    for a in range(0, 6):
        for b in range(0, 6):
            val = (2 ** a) * (3 ** b)
            if val > 0 and val <= 10000 and val not in candidates:
                candidates.append(val)
    candidates.sort()
    return candidates[:14]

def _make_value_set_2():
    """High collision values."""
    return [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]

def _make_value_set_3():
    """Geometric sequence."""
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

def _make_value_set_4():
    """Arithmetic sequence."""
    return [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

def _make_value_set_5():
    """Mixed small numbers."""
    return [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]

def _make_value_set_6():
    """Composite numbers with shared factors."""
    return [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24]

def _make_value_set_7():
    """Near-geometric with perturbations for sum collisions."""
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    perturbed = list(base)
    perturbed[1] = 3
    perturbed[3] = 10
    perturbed[5] = 40
    return perturbed

def _make_value_set_8():
    """Values designed for both sum and product collisions."""
    return [1, 2, 3, 4, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072]

def _make_value_set_9():
    """Small composites with shared factors for product collisions."""
    return [2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 25, 27]

def _make_value_set_10():
    """Mix of small primes and composites."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16]

def _make_value_set_11():
    """Even numbers with shared factors."""
    return [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

def _make_value_set_12():
    """Numbers that are products of small primes."""
    return [6, 10, 12, 14, 15, 18, 20, 21, 22, 24, 26, 28, 30, 33]

def construct():
    start_time = time.time()
    deadline = start_time + 45  # 45s deadline, 15s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    value_strategies = [
        _make_value_set_1,
        _make_value_set_2,
        _make_value_set_3,
        _make_value_set_4,
        _make_value_set_5,
        _make_value_set_6,
        _make_value_set_7,
        _make_value_set_8,
        _make_value_set_9,
        _make_value_set_10,
        _make_value_set_11,
        _make_value_set_12,
    ]
    
    for make_values in value_strategies:
        if time.time() > deadline:
            break
        
        values = make_values()
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        # Greedy edge selection
        edges = _greedy_edge_selection(values, P['m'], all_edges)
        if len(edges) != P['m']:
            continue
        
        # Fast local search on edges only
        score, edges = _fast_local_search(values, edges, all_edges, deadline, max_iters=300)
        
        # Optimize values with current edges
        if time.time() < deadline:
            score2, values2 = _optimize_values(values, edges, deadline, max_iters=100)
            if score2 < score:
                score = score2
                values = values2
        
        if score < best_score:
            best_score = score
            best_edges = [list(e) for e in edges]
            best_values = list(values)
    
    # Fallback if nothing found
    if best_edges is None:
        values1 = _make_value_set_1()
        edges = _greedy_edge_selection(values1, P['m'], all_edges)
        best_values = values1
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)
    
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
