"""Improved program: divisor-structured values + fast edge search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import Counter

_DEADLINE = time.time() + 82  # safety margin before 90s kill

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

def _generate_divisor_values():
    """Generate 14 distinct positive integers <= 10000 with rich divisor structure."""
    # Strategy: pick numbers that are divisors of a highly composite number
    # This ensures many product collisions: if a*b = c*d, we get product reuse.
    
    # Approach 1: divisors of 2^a * 3^b * 5^c * 7^d
    # Let's use numbers from the set of divisors of 2^6 * 3^4 * 5^2 * 7^1 = 64*81*25*7 = 90720
    # We need 14 distinct values <= 10000
    
    # Key insight: if we pick values that are all divisors of some N,
    # then products a*b where a|N and b|N are also divisors of N^2,
    # giving many collisions.
    
    # Even better: pick values from a multiplicative group structure.
    # Use powers of 2 and 3: {2^i * 3^j}
    
    # Let's try: all numbers of form 2^i * 3^j where i+j <= some bound
    # 2^0*3^0=1, 2^1*3^0=2, 2^0*3^1=3, 2^2*3^0=4, 2^1*3^1=6, 2^0*3^2=9,
    # 2^3*3^0=8, 2^2*3^1=12, 2^1*3^2=18, 2^0*3^3=27, 2^4*3^0=16,
    # 2^3*3^1=24, 2^2*3^2=36, 2^1*3^3=54
    # That's 14 values! All <= 10000.
    
    vals = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 54]
    assert len(vals) == 14
    assert len(set(vals)) == 14
    assert all(1 <= v <= 10000 for v in vals)
    return vals

def _generate_divisor_values_v2():
    """Alternative: more spread out divisor-structured values."""
    # Use divisors of a larger highly composite number for more product variety
    # but still with collisions
    # 2^4 * 3^3 * 5^2 * 7 = 16*27*25*7 = 75600
    # Pick 14 divisors that are spread out
    vals = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    assert len(vals) == 14
    assert len(set(vals)) == 14
    assert all(1 <= v <= 10000 for v in vals)
    return vals

def _generate_divisor_values_v3():
    """Larger spread for sum diversity control."""
    # Mix of small and medium divisor-rich numbers
    vals = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
    assert len(vals) == 14
    assert len(set(vals)) == 14
    assert all(1 <= v <= 10000 for v in vals)
    return vals

def _fast_anneal(values, n, m, edge_sp, time_budget=5.0, init_edges=None):
    """
    Fast simulated annealing on edge selection with efficient delta computation.
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    # Index all edges for fast lookup
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Initialize
    if init_edges is not None and len(init_edges) == m:
        current_edges = set(init_edges)
    else:
        current_edges = set(random.sample(all_edges, m))
    
    current_sums_set = set()
    current_prods_set = set()
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        current_sums_set.add(s)
        current_prods_set.add(p)
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    current_score = max(len(current_sums_set), len(current_prods_set))
    best_score = current_score
    best_edges = set(current_edges)
    
    T = 2.0
    T_min = 0.01
    cooling = 0.9998
    
    # Precompute for speed
    edge_sums = [edge_sp[e][0] for e in all_edges]
    edge_prods = [edge_sp[e][1] for e in all_edges]
    
    it = 0
    while time.time() - start_time < time_budget and time.time() < _DEADLINE - 3:
        if current_score <= 3:
            break
        
        it += 1
        
        # Pick random edge to remove
        remove_idx = random.randint(0, m - 1)
        remove_edge = list(current_edges)[remove_idx]
        
        # Pick random edge to add (not already in current)
        add_idx = random.randint(0, total_edges - 1)
        add_edge = all_edges[add_idx]
        
        if add_edge == remove_edge or add_edge in current_edges:
            continue
        
        rs = edge_sums[remove_idx if remove_edge in edge_index else edge_index[remove_edge]]
        rp = edge_prods[remove_idx if remove_edge in edge_index else edge_index[remove_edge]]
        as_ = edge_sums[edge_index[add_edge]]
        ap = edge_prods[edge_index[add_edge]]
        
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
                # Reset temperature slightly on improvement
                T = max(T, 1.0)
        
        T = max(T_min, T * cooling)
    
    return best_score, best_edges

def _greedy_edge_selection(values, n, m, edge_sp, time_budget=2.0):
    """
    Greedy: select edges that maximize collision (reuse of existing sums/products).
    """
    start_time = time.time()
    
    all_edges = list(combinations(range(n), 2))
    edge_sums = [edge_sp[e][0] for e in all_edges]
    edge_prods = [edge_sp[e][1] for e in all_edges]
    
    # Sort edges by how "common" their sum and product are
    sum_freq = Counter(edge_sums)
    prod_freq = Counter(edge_prods)
    
    # Score each edge: prefer edges whose sum and product are already common
    edge_priority = []
    for i, e in enumerate(all_edges):
        sf = sum_freq[edge_sums[i]]
        pf = prod_freq[edge_prods[i]]
        # Higher priority if sum and product are more frequent
        priority = sf * pf
        edge_priority.append((priority, i))
    
    edge_priority.sort(key=lambda x: -x[0])
    
    selected = set()
    selected_indices = []
    current_sums = set()
    current_prods = set()
    
    # First pass: take top priority edges
    for priority, i in edge_priority:
        if len(selected) >= m:
            break
        if time.time() - start_time > time_budget:
            break
        selected.add(i)
        selected_indices.append(i)
        current_sums.add(edge_sums[i])
        current_prods.add(edge_prods[i])
    
    # If we didn't fill m, add random
    if len(selected) < m:
        remaining = [i for i in range(len(all_edges)) if i not in selected]
        random.shuffle(remaining)
        for i in remaining[:m - len(selected)]:
            selected.add(i)
            selected_indices.append(i)
            current_sums.add(edge_sums[i])
            current_prods.add(edge_prods[i])
    
    return [all_edges[i] for i in selected_indices]

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    # Generate candidate value sets
    candidate_sets = [
        _generate_divisor_values(),
        _generate_divisor_values_v2(),
        _generate_divisor_values_v3(),
    ]
    
    # Add a few more structured sets
    # Powers and products
    vals4 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 27, 32]
    if len(set(vals4)) == 14 and all(1 <= v <= max_val for v in vals4):
        candidate_sets.append(vals4)
    
    vals5 = [2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25]
    if len(set(vals5)) == 14 and all(1 <= v <= max_val for v in vals5):
        candidate_sets.append(vals5)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidate_sets:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    num_candidates = len(unique_candidates)
    
    # Allocate time per candidate
    time_per_candidate = 75.0 / num_candidates
    
    for idx, vals in enumerate(unique_candidates):
        if time.time() > _DEADLINE - 5:
            break
        
        # Precompute edge info
        all_edges = list(combinations(range(n), 2))
        edge_sp = {}
        for e in all_edges:
            u, v = e
            edge_sp[e] = (vals[u] + vals[v], vals[u] * vals[v])
        
        # Step 1: Greedy initialization
        greedy_budget = min(1.5, time_per_candidate * 0.15)
        greedy_edges = _greedy_edge_selection(vals, n, m, edge_sp, time_budget=greedy_budget)
        
        # Step 2: Annealing starting from greedy
        anneal_budget = min(time_per_candidate * 0.75, 25.0)
        score, edges = _fast_anneal(vals, n, m, edge_sp, time_budget=anneal_budget, init_edges=greedy_edges)
        
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
