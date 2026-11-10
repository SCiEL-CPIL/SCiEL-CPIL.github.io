import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def get_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def collision_aware_edge_selection(values):
    """Select 35 edges maximizing sum and product collisions."""
    n = len(values)
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
    
    # Count how many edges share each sum and product
    sum_freq = {}
    prod_freq = {}
    for i, j, s, p in all_edges:
        sum_freq[s] = sum_freq.get(s, 0) + 1
        prod_freq[p] = prod_freq.get(p, 0) + 1
    
    # Score each edge by collision potential
    def edge_score(idx):
        i, j, s, p = all_edges[idx]
        return (sum_freq[s] + prod_freq[p])
    
    # Sort by collision potential descending
    sorted_indices = sorted(range(len(all_edges)), key=lambda x: edge_score(x), reverse=True)
    
    chosen_edges = []
    chosen_sums = set()
    chosen_prods = set()
    used = set()
    
    for idx in sorted_indices:
        if len(chosen_edges) >= P['m']:
            break
        i, j, s, p = all_edges[idx]
        chosen_edges.append([i, j])
        chosen_sums.add(s)
        chosen_prods.add(p)
        used.add(idx)
    
    return chosen_edges, max(len(chosen_sums), len(chosen_prods))

def local_search_edges(values, initial_edges):
    """Local search to improve edge selection."""
    n = len(values)
    edge_data = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_data[(i,j)] = (values[i] + values[j], values[i] * values[j])
    
    current_edges = set(tuple(e) for e in initial_edges)
    
    sum_counts = {}
    prod_counts = {}
    for e in current_edges:
        s, p = edge_data[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    distinct_sums = set(sum_counts.keys())
    distinct_prods = set(prod_counts.keys())
    current_score = max(len(distinct_sums), len(distinct_prods))
    
    all_possible_edges = [tuple(e) for e in combinations(range(n), 2)]
    
    improved = True
    while improved:
        improved = False
        best_move = None
        best_score = current_score
        
        for e_out in list(current_edges):
            s_out, p_out = edge_data[e_out]
            
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
            
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
            
            for e_in in all_possible_edges:
                if e_in in current_edges:
                    continue
                
                s_in, p_in = edge_data[e_in]
                
                new_ds = len(distinct_sums) + (0 if s_in in sum_counts else 1)
                new_dp = len(distinct_prods) + (0 if p_in in prod_counts else 1)
                
                new_score = max(new_ds, new_dp)
                
                if new_score < best_score:
                    best_score = new_score
                    best_move = (e_out, e_in)
            
            sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
            if s_out not in distinct_sums:
                distinct_sums.add(s_out)
                
            prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
            if p_out not in distinct_prods:
                distinct_prods.add(p_out)
        
        if best_move:
            e_out, e_in = best_move
            s_out, p_out = edge_data[e_out]
            s_in, p_in = edge_data[e_in]
            
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
            
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
            
            current_edges.remove(e_out)
            
            sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
            if s_in not in distinct_sums:
                distinct_sums.add(s_in)
                
            prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
            if p_in not in distinct_prods:
                distinct_prods.add(p_in)
                
            current_edges.add(e_in)
            
            current_score = best_score
            improved = True
            
    return [list(e) for e in current_edges], current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    
    # Structured value sets designed for collision potential
    value_sets = [
        # Arithmetic progressions with small differences
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        # Values with multiplicative structure
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        # Geometric-ish
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Mixed: small values for sum collisions
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        # Values where many products collide
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 25],
    ]
    
    # Filter valid sets
    valid_sets = []
    for s in value_sets:
        if len(s) == n and len(set(s)) == n and all(isinstance(x, int) and 0 < x <= 10000 for x in s):
            valid_sets.append(list(s))
    
    if not valid_sets:
        valid_sets = [list(range(1, n+1))]
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    for vals in valid_sets:
        if time.time() > deadline:
            break
        
        # Collision-aware edge selection
        edges, score = collision_aware_edge_selection(vals)
        
        # Local search refinement
        edges, score = local_search_edges(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = list(vals)
            best_edges = edges
    
    # Value perturbation local search
    if best_values is not None:
        current_values = list(best_values)
        current_score = best_score
        current_edges = best_edges
        
        rng = random.Random(42)
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve < max_no_improve:
            i = rng.randint(0, n-1)
            old = current_values[i]
            
            # Try small perturbations
            candidates = []
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10]:
                nv = old + delta
                if 0 < nv <= 10000 and nv not in current_values:
                    candidates.append(nv)
            
            # Also try random
            candidates.append(rng.randint(1, 10000))
            
            found_improvement = False
            for nv in candidates:
                if nv <= 0 or nv > 10000 or nv in current_values:
                    continue
                
                current_values[i] = nv
                edges, score = collision_aware_edge_selection(current_values)
                edges, score = local_search_edges(current_values, edges)
                
                if score < current_score:
                    current_score = score
                    current_edges = edges
                    if score < best_score:
                        best_score = score
                        best_values = list(current_values)
                        best_edges = edges
                    no_improve = 0
                    found_improvement = True
                    break
                else:
                    current_values[i] = old
            
            if not found_improvement:
                no_improve += 1
    
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, best_score = collision_aware_edge_selection(best_values)
        best_edges, best_score = local_search_edges(best_values, best_edges)
    
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
