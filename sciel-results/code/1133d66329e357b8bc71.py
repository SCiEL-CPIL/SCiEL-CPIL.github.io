import time
import random
from itertools import combinations

def get_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def greedy_edge_select(values, m=35):
    """Greedily select m edges minimizing max(distinct_sums, distinct_products)."""
    n = len(values)
    # Precompute all edge sums and products
    edge_info = {}
    for i in range(n):
        for j in range(i+1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_info[(i,j)] = (s, p)
    
    all_edges = list(edge_info.keys())
    chosen = set()
    sum_counts = {}
    prod_counts = {}
    distinct_sums = set()
    distinct_prods = set()
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        for e in all_edges:
            if e in chosen:
                continue
            s, p = edge_info[e]
            new_sums = len(distinct_sums) + (0 if s in sum_counts else 1)
            new_prods = len(distinct_prods) + (0 if p in prod_counts else 1)
            score = max(new_sums, new_prods)
            if score < best_score:
                best_score = score
                best_edge = e
        if best_edge is None:
            break
        chosen.add(best_edge)
        s, p = edge_info[best_edge]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        distinct_sums.add(s)
        distinct_prods.add(p)
    
    return list(chosen), max(len(distinct_sums), len(distinct_prods))

def local_search_edges(values, initial_edges, m=35):
    """Local search: swap edges to improve score."""
    n = len(values)
    edge_info = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_info[(i,j)] = (values[i] + values[j], values[i] * values[j])
    
    current_edges = set(tuple(e) for e in initial_edges)
    sum_counts = {}
    prod_counts = {}
    for e in current_edges:
        s, p = edge_info[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    distinct_sums = set(sum_counts.keys())
    distinct_prods = set(prod_counts.keys())
    current_score = max(len(distinct_sums), len(distinct_prods))
    
    all_possible = [tuple(e) for e in combinations(range(n), 2)]
    
    improved = True
    while improved:
        improved = False
        best_move = None
        best_score = current_score
        
        for e_out in list(current_edges):
            s_out, p_out = edge_info[e_out]
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
            
            for e_in in all_possible:
                if e_in in current_edges:
                    continue
                s_in, p_in = edge_info[e_in]
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
            s_out, p_out = edge_info[e_out]
            s_in, p_in = edge_info[e_in]
            
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
    n = 14
    m = 35
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Generate structured value sets
    candidate_sets = []
    
    # Multiples of small bases to create product/sum structure
    for base in range(1, 20):
        for start in range(1, 15):
            vals = [base * (start + i) for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidate_sets.append(vals)
    
    # Geometric progressions
    for start in range(1, 10):
        for ratio in [2, 3]:
            vals = [start * (ratio ** i) for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidate_sets.append(vals)
    
    # Small integers
    candidate_sets.append(list(range(1, n+1)))
    candidate_sets.append(list(range(2, n+2)))
    
    # Multiples with shared factors
    for base in [6, 12, 18, 24, 30, 36]:
        for start in range(1, 10):
            vals = [base * (start + i) for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidate_sets.append(vals)
    
    # Random structured
    rng = random.Random(42)
    for _ in range(50):
        base = rng.randint(1, 30)
        start = rng.randint(1, 20)
        vals = [base * (start + i) + rng.randint(-base//2, base//2) for i in range(n)]
        vals = sorted(set(max(1, min(10000, v)) for v in vals))
        while len(vals) < n:
            vals.append(rng.randint(1, 10000))
        vals = vals[:n]
        if len(set(vals)) == n and all(1 <= v <= 10000 for v in vals):
            candidate_sets.append(vals)
    
    # Evaluate each candidate
    for vals in candidate_sets:
        if time.time() > deadline:
            break
        edges, score = greedy_edge_select(vals, m)
        edges, score = local_search_edges(vals, edges, m)
        if score < best_score:
            best_score = score
            best_values = list(vals)
            best_edges = edges
    
    # Local search on values
    if best_values is not None:
        current_values = list(best_values)
        current_score = best_score
        current_edges = best_edges
        rng = random.Random(123)
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve < max_no_improve:
            i = rng.randint(0, n-1)
            old_val = current_values[i]
            new_val = rng.randint(1, 10000)
            if new_val in current_values:
                no_improve += 1
                continue
            
            current_values[i] = new_val
            edges, score = greedy_edge_select(current_values, m)
            edges, score = local_search_edges(current_values, edges, m)
            
            if score < current_score:
                current_score = score
                current_edges = edges
                if score < best_score:
                    best_score = score
                    best_values = list(current_values)
                    best_edges = edges
                no_improve = 0
            else:
                current_values[i] = old_val
                no_improve += 1
    
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, best_score = greedy_edge_select(best_values, m)
        best_edges, best_score = local_search_edges(best_values, best_edges, m)
    
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
