"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict, Counter

import time
import random
import math
from itertools import combinations
from collections import defaultdict, Counter

_DEADLINE = time.time() + 85  # safety margin before 90s kill

def _score_detailed(values, edges):
    """Return (sums_set, prods_set, score)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return sums, prods, max(len(sums), len(prods))

def _fast_anneal(values, n, m, edge_sp, time_budget=5.0):
    """
    Fast simulated annealing on edge selection.
    Avoids O(N) candidate filtering by using rejection sampling or direct random pick from all edges.
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    # Initialize with greedy: pick edges with highest collision potential
    # For speed, we just pick random initially or a simple greedy if time permits
    # Let's do a quick greedy init: sort by (sum_count + prod_count) is expensive to compute globally.
    # Instead, just start with a random set of m edges.
    current_edges = set(random.sample(all_edges, m))
    
    current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
    best_score = current_score
    best_edges = set(current_edges)
    
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    max_iter = 50000 # Fixed high number, time checked inside
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 2:
            break
        if current_score <= 3:
            break
        
        # Pick random edge to remove
        remove_edge = random.choice(list(current_edges))
        
        # Pick random edge to add (from all edges)
        # If it's the same or already in set, we just skip or pick again.
        # To avoid infinite loops, we just pick a random index.
        add_edge = all_edges[random.randint(0, total_edges - 1)]
        
        if add_edge == remove_edge:
            continue
            
        if add_edge in current_edges:
            # If we pick an edge already in the set, it's not a valid swap (no change in set size)
            # We can either skip or treat it as a null move. Skipping is safer.
            continue

        rs, rp = edge_sp[remove_edge]
        as_, ap = edge_sp[add_edge]
        
        # Compute delta
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
        
        new_num_sums = len(current_sums_set) + delta_sums
        new_num_prods = len(current_prods_set) + delta_prods
        new_score = max(new_num_sums, new_num_prods)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or (T > T_min and random.random() < math.exp(-delta_score / T)):
            # Apply move
            current_edges.discard(remove_edge)
            current_edges.add(add_edge)
            
            # Update counts
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
            
            sum_counts[as_] += 1
            prod_counts[ap] += 1
            
            # Update sets
            if sum_counts[rs] == 0:
                current_sums_set.discard(rs)
            if sum_counts[as_] == 1:
                current_sums_set.add(as_)
            
            if prod_counts[rp] == 0:
                current_prods_set.discard(rp)
            if prod_counts[ap] == 1:
                current_prods_set.add(ap)
            
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        
        T = max(T_min, T * cooling)
    
    return best_score, best_edges

def _generate_value_candidates(n, max_val=10000):
    """Generate a smaller, more focused set of high-quality candidates."""
    candidates = []
    
    # 1. Consecutive small integers (good for sums)
    candidates.append(list(range(1, n + 1)))
    
    # 2. Smooth numbers (highly composite, good for products)
    # 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    smooth_14 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    if len(smooth_14) == n:
        candidates.append(smooth_14)
        
    # 3. Another smooth set
    smooth_14b = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
    if len(smooth_14b) == n:
        candidates.append(smooth_14b)
        
    # 4. Powers of 2 (extreme product structure, bad sum structure)
    candidates.append([2**i for i in range(n)])
    
    # 5. Random small numbers (diversity)
    random.seed(42)
    for _ in range(10):
        vals = sorted(random.sample(range(1, 40), n))
        if len(set(vals)) == n:
            candidates.append(vals)
            
    # 6. Geometric progression
    base = 1.5
    vals = [max(1, int(base**i)) for i in range(n)]
    # Ensure distinct
    seen = set()
    unique_vals = []
    for v in vals:
        while v in seen:
            v += 1
        seen.add(v)
        unique_vals.append(v)
    if len(unique_vals) == n and all(v <= max_val for v in unique_vals):
        candidates.append(sorted(unique_vals))

    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    candidates = _generate_value_candidates(n, max_val)
    
    # We don't sort by complex heuristic to save time. 
    # We'll just iterate through them.
    # To ensure we use time efficiently, we allocate a fixed budget per candidate.
    # Total time budget: 80s.
    # If we have 15 candidates, ~5s each.
    
    num_candidates = len(candidates)
    time_per_candidate = 80.0 / num_candidates
    
    for idx, vals in enumerate(candidates):
        if time.time() > _DEADLINE - 5:
            break
        
        # Precompute edge info
        all_edges = list(combinations(range(n), 2))
        edge_sp = {}
        for e in all_edges:
            u, v = e
            edge_sp[e] = (vals[u] + vals[v], vals[u] * vals[v])
        
        # Run annealing with strict time limit
        # Use slightly less time than allocated to account for overhead
        budget = time_per_candidate * 0.9
        score, edges = _fast_anneal(vals, n, m, edge_sp, time_budget=budget)
        
        if score < best_score:
            best_score = score
            best_result = (vals, [list(e) for e in edges])
            if best_score <= 3:
                break
    
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
