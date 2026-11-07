"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
DEADLINE = 85.0  # safety margin before 90s kill

def _raw_score(values, edges):
    """Compute the raw objective: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _greedy_edge_select(values, target_m):
    """
    Greedily select target_m edges to minimize the growth in distinct sums/products.
    At each step, pick the edge that adds the fewest NEW distinct values to the union.
    """
    all_edges = list(combinations(range(N), 2))
    current_sums = set()
    current_prods = set()
    selected = []
    
    # Precompute sums and products for all edges
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_sums[(u, v)] = a + b
        edge_prods[(u, v)] = a * b
    
    remaining = list(all_edges)
    
    for _ in range(target_m):
        best_edge = None
        best_score = float('inf')
        best_new_sums = 0
        best_new_prods = 0
        
        for e in remaining:
            s = edge_sums[e]
            p = edge_prods[e]
            new_s = 0 if s in current_sums else 1
            new_p = 0 if p in current_prods else 1
            # We want to minimize max(len(sums)+new_s, len(prods)+new_p)
            # But since we're adding one at a time, minimize the increase
            # Actually, we should minimize the resulting max
            cur_s_count = len(current_sums) + new_s
            cur_p_count = len(current_prods) + new_p
            score = max(cur_s_count, cur_p_count)
            
            if score < best_score:
                best_score = score
                best_edge = e
                best_new_sums = new_s
                best_new_prods = new_p
            elif score == best_score:
                # Tie-break: prefer edges that add fewer total new values
                if best_new_sums + best_new_prods > new_s + new_p:
                    best_edge = e
                    best_new_sums = new_s
                    best_new_prods = new_p
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_sums.add(edge_sums[best_edge])
        current_prods.add(edge_prods[best_edge])
        selected.append([u, v])
        remaining.remove(best_edge)
    
    return selected

def _try_value_set(values):
    """Given a set of values, greedily select edges and return the score."""
    edges = _greedy_edge_select(values, M)
    if len(edges) < M:
        return float('inf')
    return _raw_score(values, edges)

def _local_search(values, time_budget):
    """
    Try local perturbations of the value set: swap, add/subtract small deltas.
    """
    best_values = list(values)
    best_score = _try_value_set(best_values)
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        new_values = list(best_values)
        # Random perturbation
        idx = random.randint(0, N - 1)
        delta = random.choice([-3, -2, -1, 1, 2, 3, -10, 10, -20, 20])
        new_val = new_values[idx] + delta
        if new_val < 1 or new_val > MAX_VAL:
            continue
        if new_val in new_values:
            continue
        new_values[idx] = new_val
        
        score = _try_value_set(new_values)
        if score < best_score:
            best_values = new_values
            best_score = score
    
    return best_values, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_values = None
    best_score = float('inf')
    
    # Strategy 1: Start with a structured set - use multiples of a base
    # to create many collisions in sums and products
    # Try various base values
    
    candidate_sets = []
    
    # Set A: Arithmetic progression with small step
    candidate_sets.append(list(range(1, N + 1)))
    
    # Set B: All even numbers
    candidate_sets.append([2 * i for i in range(1, N + 1)])
    
    # Set C: Powers of 2 (but keep <= 10000)
    p2 = [1]
    for i in range(1, 14):
        p2.append(p2[-1] * 2)
    if max(p2) <= MAX_VAL:
        candidate_sets.append(p2)
    
    # Set D: Mix of small numbers with some structure
    candidate_sets.append([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    
    # Set E: Geometric-like with base 3
    g3 = [3**i for i in range(N)]
    if max(g3) <= MAX_VAL:
        candidate_sets.append(g3)
    
    # Set F: Repetitive structure - pairs of equal-ish values
    # But they must be distinct, so use close values
    candidate_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Set G: Values that are divisors of a common number (LCM-friendly)
    # e.g., divisors of 2520 = 2^3 * 3^2 * 5 * 7
    divs_2520 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18]
    candidate_sets.append(divs_2520)
    
    # Set H: Fibonacci-like
    fib = [1, 1]
    for i in range(N - 2):
        fib.append(fib[-1] + fib[-2])
    # Make distinct
    fib_distinct = []
    for v in fib:
        if v not in fib_distinct:
            fib_distinct.append(v)
    while len(fib_distinct) < N:
        fib_distinct.append(fib_distinct[-1] + fib_distinct[-2])
    if max(fib_distinct) <= MAX_VAL:
        candidate_sets.append(fib_distinct)
    
    # Set I: Small numbers clustered
    candidate_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Set J: Multiples of 1,2,3,... with some scaling
    candidate_sets.append([i * 7 for i in range(1, N + 1)])
    
    # Set K: Mix - some small, some medium
    candidate_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512])
    
    # Set L: All odd small numbers
    candidate_sets.append([2*i+1 for i in range(N)])
    
    # Set M: Values from 1 to 14 but with one large value
    candidate_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100])
    
    # Evaluate each candidate
    for vals in candidate_sets:
        if time.time() > deadline - 5:
            break
        if len(vals) != N:
            continue
        if len(set(vals)) != N:
            continue
        if any(v < 1 or v > MAX_VAL for v in vals):
            continue
        score = _try_value_set(vals)
        if score < best_score:
            best_score = score
            best_values = list(vals)
    
    # Now do local search from the best candidate
    if best_values is not None and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 10
        best_values, best_score = _local_search(best_values, remaining_time)
    
    # Final edge selection
    edges = _greedy_edge_select(best_values, M)
    
    # Ensure valid types
    result_values = [int(v) for v in best_values]
    result_edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': result_values, 'edges': result_edges}
# EVOLVE-BLOCK-END
