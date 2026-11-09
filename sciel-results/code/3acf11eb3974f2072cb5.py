import time
import random
import math
from itertools import combinations
from collections import Counter

_DEADLINE = time.time() + 55  # Safety margin

def _score_edges(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _optimize_edges_for_values(values, time_budget):
    n = len(values)
    m = 35
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    # Precompute sum and product for each edge
    edge_sum = {}
    edge_prod = {}
    for u, v in all_edges:
        edge_sum[(u, v)] = values[u] + values[v]
        edge_prod[(u, v)] = values[u] * values[v]

    # Greedy Initialization
    # Start with empty set, add edge that minimizes max(|sums|, |prods|)
    current_edges = set()
    sum_counts = Counter()
    prod_counts = Counter()
    
    # To speed up greedy, we can just pick a few random edges first, then refine
    # Or just do a simple greedy:
    # For each step, try all remaining edges and pick the one that results in lowest cost.
    # This is O(m * E * log(E)) which is fast enough for 35 edges.
    
    remaining_edges = set(all_edges)
    
    for _ in range(m):
        best_edge = None
        best_cost = float('inf')
        
        # Sample if too many candidates? No, 91 is small.
        for e in remaining_edges:
            s = edge_sum[e]
            p = edge_prod[e]
            
            # Simulate adding e
            new_sum_count = 1 if sum_counts[s] == 0 else 0
            new_prod_count = 1 if prod_counts[p] == 0 else 0
            
            new_distinct_sums = len(sum_counts) + new_sum_count
            new_distinct_prods = len(prod_counts) + new_prod_count
            
            cost = max(new_distinct_sums, new_distinct_prods)
            
            if cost < best_cost:
                best_cost = cost
                best_edge = e
                
        # Add best edge
        if best_edge is not None:
            current_edges.add(best_edge)
            remaining_edges.remove(best_edge)
            s = edge_sum[best_edge]
            p = edge_prod[best_edge]
            sum_counts[s] += 1
            prod_counts[p] += 1

    # Current state
    current_score = max(len(sum_counts), len(prod_counts))
    best_score = current_score
    best_edges = set(current_edges)
    
    # Simulated Annealing
    start_time = time.time()
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    # Indices for all_edges for faster random access
    edge_indices = {e: i for i, e in enumerate(all_edges)}
    
    while time.time() - start_time < time_budget:
        if time.time() > _DEADLINE:
            break
            
        # Pick random edge to remove
        remove_edge = random.choice(list(current_edges))
        
        # Pick random edge to add
        add_edge = random.choice(all_edges)
        
        if add_edge in current_edges:
            continue
            
        # Calculate delta
        rs = edge_sum[remove_edge]
        rp = edge_prod[remove_edge]
        as_ = edge_sum[add_edge]
        ap = edge_prod[add_edge]
        
        delta_sums = 0
        if sum_counts[rs] == 1:
            delta_sums -= 1
        if sum_counts[as_] == 0:
            delta_sums += 1
            
        delta_prods = 0
        if prod_counts[rp] == 1:
            delta_prods -= 1
        if prod_counts[ap] == 0:
            delta_prods += 1
            
        new_distinct_sums = len(sum_counts) + delta_sums
        new_distinct_prods = len(prod_counts) + delta_prods
        new_score = max(new_distinct_sums, new_distinct_prods)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or (T > T_min and random.random() < math.exp(-delta_score / T)):
            # Accept move
            current_edges.discard(remove_edge)
            current_edges.add(add_edge)
            
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
                
            sum_counts[as_] += 1
            prod_counts[ap] += 1
            
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                if best_score <= 7: # Heuristic lower bound?
                    # We can stop early if we hit a very good score
                    pass # Don't break immediately, might get better, but 7 is likely close to optimal
                
        T = max(T_min, T * cooling)
        
    return best_score, best_edges

def construct():
    n = 14
    m = 35
    
    # Candidate value sets
    candidates = [
        list(range(1, 15)),
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ]
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    best_score = m
    best_result = None
    
    # Time allocation
    num_candidates = len(unique_candidates)
    # 50s total budget.
    # Allocate more time to promising candidates?
    # Let's just divide evenly, but give more to the first few.
    # Or just run SA for a fixed time per candidate.
    # 50s / 20 candidates = 2.5s per candidate.
    
    for vals in unique_candidates:
        if time.time() > _DEADLINE - 5:
            break
            
        # Ensure values are distinct and positive
        if len(set(vals)) != 14:
            continue
        if any(v <= 0 for v in vals):
            continue
        if any(v > 10000 for v in vals):
            continue
            
        # Optimize edges
        score, edges = _optimize_edges_for_values(vals, time_budget=2.0)
        
        if score < best_score:
            best_score = score
            best_result = (vals, edges)
            if best_score <= 8:
                # We found a very good solution, use remaining time to polish this one
                # Re-run SA with more time on this specific value set
                pass # Handled by loop logic if we break? No, let's just continue or break if time is low.
                
    if best_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    else:
        values, edges = best_result
        
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
