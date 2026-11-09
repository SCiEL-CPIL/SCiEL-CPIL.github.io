import time
import random
from itertools import combinations

_DEADLINE = time.time() + 85.0

def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _optimize_edges(values, m, n, time_budget=1.0):
    """
    Optimizes edge selection for given values.
    Uses greedy initialization and local search.
    """
    all_edges = list(combinations(range(n), 2))
    E = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [0] * E
    edge_prods = [0] * E
    edge_map = {} # index -> (u, v)
    for i, (u, v) in enumerate(all_edges):
        edge_sums[i] = values[u] + values[v]
        edge_prods[i] = values[u] * values[v]
        edge_map[i] = (u, v)

    # State representation:
    # current_edges: set of edge indices
    # sum_counts: dict sum_val -> count
    # prod_counts: dict prod_val -> count
    # distinct_sums, distinct_prods
    
    def get_score(distinct_sums, distinct_prods):
        return max(distinct_sums, distinct_prods)

    # 1. Greedy Initialization
    # We want to pick m edges that minimize the growth of distinct sums/prods.
    # We simulate adding edges one by one.
    
    # To make it fast, we don't re-evaluate all remaining edges from scratch every time,
    # but for N=14, E=91, M=35, it's cheap enough to just check candidates.
    
    current_edges = set()
    sum_counts = {}
    prod_counts = {}
    distinct_sums = 0
    distinct_prods = 0
    
    remaining_edges = set(range(E))
    
    for _ in range(m):
        best_edge = -1
        best_delta = float('inf')
        best_next_s = 0
        best_next_p = 0
        
        for e in remaining_edges:
            s = edge_sums[e]
            p = edge_prods[e]
            
            new_ds = distinct_sums + (1 if sum_counts.get(s, 0) == 0 else 0)
            new_dp = distinct_prods + (1 if prod_counts.get(p, 0) == 0 else 0)
            
            # Objective: minimize max(new_ds, new_dp)
            # Tie-breaker: minimize sum of increases
            score = max(new_ds, new_dp)
            delta = score
            
            # We want to pick the edge that leads to the lowest score
            # If scores are equal, prefer the one that adds less "load" (counts)
            # Actually, just picking min score is a good heuristic.
            
            if delta < best_delta:
                best_delta = delta
                best_edge = e
                best_next_s = new_ds
                best_next_p = new_dp
        
        if best_edge == -1:
            break
            
        current_edges.add(best_edge)
        remaining_edges.remove(best_edge)
        
        s = edge_sums[best_edge]
        p = edge_prods[best_edge]
        
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
        distinct_sums = best_next_s
        distinct_prods = best_next_p
        
    current_score = get_score(distinct_sums, distinct_prods)
    best_score = current_score
    best_edges_set = set(current_edges)
    
    # 2. Local Search
    # We try to swap an edge in current_edges with an edge in remaining_edges.
    # We use a list for current_edges for fast random access? 
    # Set is fine for membership, list for iteration if needed.
    
    in_list = list(current_edges)
    out_list = list(remaining_edges)
    
    if not out_list:
        return best_score, [list(edge_map[e]) for e in best_edges_set]
        
    start_time = time.time()
    iterations = 0
    max_iterations = 50000
    
    while iterations < max_iterations:
        if time.time() > start_time + time_budget:
            break
            
        iterations += 1
        
        # Randomly pick one to remove and one to add
        idx_in = random.randint(0, len(in_list) - 1)
        e_in = in_list[idx_in]
        
        idx_out = random.randint(0, len(out_list) - 1)
        e_out = out_list[idx_out]
        
        s_in = edge_sums[e_in]
        p_in = edge_prods[e_in]
        s_out = edge_sums[e_out]
        p_out = edge_prods[e_out]
        
        # Calculate potential new distinct counts
        # Remove e_in
        ns_sums = distinct_sums
        if sum_counts[s_in] == 1:
            ns_sums -= 1
            
        ns_prods = distinct_prods
        if prod_counts[p_in] == 1:
            ns_prods -= 1
            
        # Add e_out
        if sum_counts.get(s_out, 0) == 0:
            ns_sums += 1
        if prod_counts.get(p_out, 0) == 0:
            ns_prods += 1
            
        new_score = get_score(ns_sums, ns_prods)
        
        if new_score <= current_score:
            # Accept
            # Update counts
            sum_counts[s_in] -= 1
            if sum_counts[s_in] == 0:
                del sum_counts[s_in]
            prod_counts[p_in] -= 1
            if prod_counts[p_in] == 0:
                del prod_counts[p_in]
                
            sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
            prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
            
            distinct_sums = ns_sums
            distinct_prods = ns_prods
            current_score = new_score
            
            # Update sets/lists
            current_edges.discard(e_in)
            current_edges.add(e_out)
            remaining_edges.discard(e_out)
            remaining_edges.add(e_in)
            
            in_list[idx_in] = e_out
            out_list[idx_out] = e_in
            
            if current_score < best_score:
                best_score = current_score
                best_edges_set = set(current_edges)
                
            # If we hit a very low score, we can break early? 
            # Theoretical min for sums is 2N-3 = 25. For prods, it's harder.
            # If best_score <= 25, it's likely optimal or near-optimal.
            if best_score <= 25:
                 # We might still find lower if prods drop, but sums are bounded by 25 for APs?
                 # Actually, distinct sums can be less if we don't use all pairs.
                 # But 25 is the max possible distinct sums for ANY 14 distinct numbers?
                 # No, for distinct numbers, min distinct sums is N-1 (if we only pick adjacent?).
                 # Wait, if we pick a subset of edges, we can have fewer distinct sums.
                 # E.g. a star graph has N-1 distinct sums (center + leaf).
                 # So min distinct sums can be as low as 13.
                 # So we shouldn't stop at 25.
                 pass

    return best_score, [list(edge_map[e]) for e in best_edges_set]

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    # Candidate value sets
    candidates = []
    
    # 1. Small APs
    for start in range(1, 10):
        for step in range(1, 5):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val:
                candidates.append(vals)
                
    # 2. Specific sets
    candidates.append(list(range(1, n + 1)))
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # 3. Random small sets
    random.seed(42)
    for _ in range(50):
        # Sample from 1..50
        vals = sorted(random.sample(range(1, 51), n))
        candidates.append(vals)
        
    # 4. "Clustered" sets: Two groups of close numbers
    for _ in range(20):
        base1 = random.randint(1, 10)
        base2 = random.randint(20, 30)
        vals = []
        for i in range(7):
            vals.append(base1 + i)
        for i in range(7):
            vals.append(base2 + i)
        if len(set(vals)) == n and max(vals) <= max_val:
            candidates.append(sorted(vals))

    # Time management
    total_time = 80.0
    time_per_candidate = total_time / len(candidates)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
            
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
            
        # Run optimization
        # Allocate slightly more time if we are early, less if late?
        # Just use a fixed slice.
        score, edges = _optimize_edges(vals, m, n, time_budget=time_per_candidate)
        
        if score < best_score:
            best_score = score
            best_result = (vals, edges)
            
        # If we find a very good score, we can spend more time refining it?
        # For now, keep it simple.
        
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
        
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
