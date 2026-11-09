"""
Strategy:
1. Use value sets based on divisors of highly composite numbers (e.g., 2520, 5040).
   This creates many pairs (a,b) such that a*b = a'*b' (product collisions).
   Since the objective is max(distinct_sums, distinct_prods), reducing distinct products
   is crucial because the product space grows quadratically with value magnitude,
   while sums grow linearly. Small values with high divisor structure are ideal.
2. For a fixed value set, use a greedy algorithm to select 35 edges that maximize
   collisions. We prioritize edges that share a sum OR product with already selected edges.
3. Perform a fast local search (edge swaps) to polish the solution.
4. Repeat for a few different highly-composite-based value sets.
"""

import time
import random
from itertools import combinations

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _score_detailed(values, edges):
    """Returns (score, num_sums, num_prods)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods)), len(sums), len(prods)

def _get_divisor_values(base, count=14):
    """
    Get 'count' distinct divisors of 'base'.
    We want small divisors to keep sums small, but enough structure for product collisions.
    """
    divs = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divs.append(i)
            if i != base // i:
                divs.append(base // i)
    divs = sorted(set(divs))
    if len(divs) < count:
        return None
    # Pick the smallest 'count' divisors? Or a mix?
    # Smallest divisors give smaller sums.
    # Let's try taking the first 'count' smallest divisors.
    return divs[:count]

def _greedy_edge_selection(values, num_edges=35):
    """
    Greedily select edges to maximize collision with previously selected edges.
    Score of an edge (u,v) given current set S:
    +1 if sum(u,v) in sums(S)
    +1 if prod(u,v) in prods(S)
    Ties broken by smaller sum (to keep sum range tight) or random.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Pre-compute sum and product for all edges
    edge_data = {}
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data[(u, v)] = (s, p)
    
    selected_edges = []
    current_sums = set()
    current_prods = set()
    
    # First edge: pick the one with the smallest product (to start in dense area)
    # Or just pick the one with smallest sum?
    # Let's pick the one that has the most potential for future collisions?
    # Simple start: smallest product.
    first_edge = min(all_edges, key=lambda e: edge_data[e][1])
    selected_edges.append(first_edge)
    current_sums.add(edge_data[first_edge][0])
    current_prods.add(edge_data[first_edge][1])
    
    remaining_edges = [e for e in all_edges if e != first_edge]
    
    for _ in range(num_edges - 1):
        if not remaining_edges:
            break
            
        best_edge = None
        best_score = -1
        best_sum = float('inf')
        
        # Sample a subset if too many edges to evaluate? No, 91 edges is small.
        for e in remaining_edges:
            s, p = edge_data[e]
            score = 0
            if s in current_sums:
                score += 1
            if p in current_prods:
                score += 1
            
            # Tie breaker: prefer smaller sum to keep sum space compact
            if score > best_score or (score == best_score and s < best_sum):
                best_score = score
                best_edge = e
                best_sum = s
                
        if best_edge is None:
            break
            
        selected_edges.append(best_edge)
        current_sums.add(edge_data[best_edge][0])
        current_prods.add(edge_data[best_edge][1])
        remaining_edges.remove(best_edge)
        
    return selected_edges

def _local_search_edges(values, edges, time_limit=2.0):
    """
    Local search to improve edge selection for fixed values.
    Try swapping one edge for another if it reduces the score.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    edges_set = set(edges)
    
    current_score, _, _ = _score_detailed(values, edges_set)
    best_score = current_score
    best_edges = set(edges_set)
    
    start_time = time.time()
    
    # Pre-compute edge data
    edge_data = {}
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data[(u, v)] = (s, p)
        
    # Current sums/prods sets for fast lookup
    cur_sums = set()
    cur_prods = set()
    for e in edges_set:
        s, p = edge_data[e]
        cur_sums.add(s)
        cur_prods.add(p)
        
    # Count frequencies for quick removal logic? 
    # Actually, recalculating score from scratch is O(35) which is very fast.
    # Let's just recalculate for simplicity and correctness, 35 edges is tiny.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle to avoid local minima bias
        shuffled_edges = list(edges_set)
        random.shuffle(shuffled_edges)
        
        for e_out in shuffled_edges:
            # Try to find a better e_in
            # Sample a few random candidates for e_in
            for _ in range(10):
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                    
                # Tentative swap
                new_edges_set = set(edges_set)
                new_edges_set.remove(e_out)
                new_edges_set.add(e_in)
                
                new_score, _, _ = _score_detailed(values, new_edges_set)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges_set
                    edges_set = new_edges_set
                    current_score = new_score
                    improved = True
                    break # Restart loop with new edges
            if improved:
                break
                
        if not improved:
            # Try random swaps even if no immediate improvement found in this pass?
            # Or just break if no improvement in a full pass?
            # Let's do a few random moves to escape.
            if random.random() < 0.5:
                e_out = random.choice(list(edges_set))
                e_in = random.choice(all_edges)
                if e_in not in edges_set:
                    new_edges_set = set(edges_set)
                    new_edges_set.remove(e_out)
                    new_edges_set.add(e_in)
                    new_score, _, _ = _score_detailed(values, new_edges_set)
                    if new_score <= best_score: # Allow neutral moves
                        edges_set = new_edges_set
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_edges = new_edges_set
            else:
                break
                
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_global_score = float('inf')
    best_global_result = None
    
    # Generate candidate value sets based on highly composite numbers
    # 2520 is LCM(1..10), has many divisors.
    # 5040 is LCM(1..10)*2.
    # 720720 is LCM(1..16).
    candidates = [
        2520,
        5040,
        7560,
        10080,
        15120,
        20160,
        27720,
        45360,
        55440,
        83160,
        110880,
        166320,
        221760,
        277200,
        332640,
        498960,
        554400,
        665280,
        720720,
        1081080
    ]
    
    random.shuffle(candidates)
    
    value_sets = []
    for base in candidates:
        vals = _get_divisor_values(base, 14)
        if vals and len(set(vals)) == 14:
            # Ensure distinct and positive
            if all(v > 0 for v in vals):
                value_sets.append(vals)
                
    # Also add the simple 1..14 as a baseline
    value_sets.append(list(range(1, 15)))
    value_sets.append(list(range(2, 16)))
    
    # Deduplicate value sets (sorted tuples)
    unique_value_sets = []
    seen = set()
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen:
            seen.add(key)
            unique_value_sets.append(vs)
            
    random.shuffle(unique_value_sets)
    
    for vals in unique_value_sets:
        if time.time() > deadline - 5.0:
            break
            
        # 1. Greedy edge selection
        edges = _greedy_edge_selection(vals, 35)
        
        # 2. Local search polish
        # Allocate time for local search
        remaining = deadline - time.time()
        ls_time = min(3.0, remaining / 2.0)
        
        if ls_time > 0.1:
            best_edges, best_score = _local_search_edges(vals, edges, ls_time)
        else:
            best_edges, best_score = set(edges), _raw_score(vals, edges)
            
        if best_score < best_global_score:
            best_global_score = best_score
            best_global_result = {
                'values': list(vals),
                'edges': [list(e) for e in best_edges]
            }
            
            # Early stop if we find a very good solution
            if best_global_score <= 12:
                break

    if best_global_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = _greedy_edge_selection(vals, 35)
        best_global_result = {
            'values': vals,
            'edges': [list(e) for e in edges]
        }
        
    # Final type checking
    res_values = [int(v) for v in best_global_result['values']]
    res_edges = [[int(u), int(v)] for u, v in best_global_result['edges']]
    
    return {
        'values': res_values,
        'edges': res_edges
    }
