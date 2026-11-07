"""
Structural approach: use divisors of highly composite numbers for bounded products,
then SA for edge selection and local search for value refinement.
"""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import Counter
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}


def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _all_edges(n):
    return list(combinations(range(n), 2))


def _divisors_of(n):
    divs = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
    return sorted(divs)


def _make_candidate_values():
    """Generate diverse candidate value sets from highly composite divisors."""
    candidates = []
    
    # Highly composite / superabundant numbers
    hcn_list = [2520, 5040, 7560, 10080, 15120, 27720, 270270, 360360, 720720, 1441440, 36756720]
    
    for hcn in hcn_list:
        divs = [d for d in _divisors_of(hcn) if d <= 10000]
        if len(divs) >= 14:
            # Take first 14 (smallest)
            candidates.append(sorted(divs[:14]))
            # Take evenly spaced
            step = len(divs) // 14
            sampled = [divs[i * step] for i in range(14)]
            if len(set(sampled)) == 14:
                candidates.append(sorted(sampled))
            # Take largest 14
            candidates.append(sorted(divs[-14:]))
            # Random sample
            for _ in range(3):
                sampled = random.sample(divs, 14)
                candidates.append(sorted(sampled))
    
    # Powers of 2
    pow2 = [2**i for i in range(14)]
    candidates.append(pow2)
    
    # Powers of 3
    pow3 = [3**i for i in range(14)]
    pow3 = [v for v in pow3 if v <= 10000]
    if len(pow3) >= 14:
        candidates.append(pow3[:14])
    
    # Mixed: 2^a * 3^b
    mixed = []
    for a in range(0, 8):
        for b in range(0, 8):
            val = (2**a) * (3**b)
            if 0 < val <= 10000:
                mixed.append(val)
    mixed.sort()
    if len(mixed) >= 14:
        candidates.append(mixed[:14])
        # Random sample from mixed
        for _ in range(5):
            sampled = random.sample(mixed, 14)
            candidates.append(sorted(sampled))
    
    # Small composites with shared factors
    small_comps = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60]
    if len(small_comps) >= 14:
        candidates.append(small_comps[:14])
        for _ in range(5):
            sampled = random.sample(small_comps, 14)
            candidates.append(sorted(sampled))
    
    # Even numbers
    evens = list(range(2, 30, 2))
    if len(evens) >= 14:
        candidates.append(evens[:14])
    
    # Multiples of 6
    mult6 = [6*i for i in range(1, 15)]
    candidates.append(mult6)
    
    # Multiples of 4
    mult4 = [4*i for i in range(1, 15)]
    candidates.append(mult4)
    
    # Arithmetic progressions
    for start in [1, 2, 3, 4, 5]:
        for step in [3, 4, 5, 6, 7, 8]:
            vals = [start + i*step for i in range(14)]
            if all(0 < v <= 10000 for v in vals):
                candidates.append(vals)
    
    # Geometric with small ratio
    for base in [1, 2, 3]:
        for ratio in [2, 3, 4]:
            vals = []
            v = base
            while len(vals) < 14 and v <= 10000:
                vals.append(v)
                v *= ratio
            if len(vals) >= 14:
                candidates.append(vals[:14])
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen and len(set(c)) == 14 and all(0 < v <= 10000 for v in c):
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates


def _sa_edge_selection(values, all_edges, deadline, num_restarts=5, steps_per_restart=50000):
    """
    Simulated annealing to select 35 edges minimizing max(distinct_sums, distinct_products).
    """
    n = len(values)
    m = P['m']
    
    # Precompute edge attributes
    edge_sums = []
    edge_prods = []
    for i, j in all_edges:
        a, b = values[i], values[j]
        edge_sums.append(a + b)
        edge_prods.append(a * b)
    
    num_total = len(all_edges)
    best_score = float('inf')
    best_edges = None
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        # Random initial set
        init_indices = random.sample(range(num_total), m)
        current_edges = set(init_indices)
        
        sum_counts = Counter()
        prod_counts = Counter()
        active_sums = set()
        active_prods = set()
        
        for idx in current_edges:
            s = edge_sums[idx]
            p = edge_prods[idx]
            sum_counts[s] += 1
            prod_counts[p] += 1
            active_sums.add(s)
            active_prods.add(p)
        
        current_score = max(len(active_sums), len(active_prods))
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(all_edges[idx]) for idx in current_edges]
        
        # SA
        T = 1.0
        cooling = 0.9997
        min_T = 0.01
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 1:
                break
            
            e_out = random.choice(list(current_edges))
            e_in = random.randrange(num_total)
            if e_in in current_edges:
                continue
            
            s_out = edge_sums[e_out]
            p_out = edge_prods[e_out]
            s_in = edge_sums[e_in]
            p_in = edge_prods[e_in]
            
            # Remove
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                active_sums.discard(s_out)
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                active_prods.discard(p_out)
            
            # Add
            sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
            if sum_counts[s_in] == 1:
                active_sums.add(s_in)
            prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
            if prod_counts[p_in] == 1:
                active_prods.add(p_in)
            
            new_score = max(len(active_sums), len(active_prods))
            
            if new_score <= current_score:
                current_edges.discard(e_out)
                current_edges.add(e_in)
                current_score = new_score
            else:
                if random.random() < math.exp(-(new_score - current_score) / T):
                    current_edges.discard(e_out)
                    current_edges.add(e_in)
                    current_score = new_score
                else:
                    # Revert
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        active_sums.discard(s_in)
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        active_prods.discard(p_in)
                    
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    if sum_counts[s_out] == 1:
                        active_sums.add(s_out)
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                    if prod_counts[p_out] == 1:
                        active_prods.add(p_out)
            
            T = max(min_T, T * cooling)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(all_edges[idx]) for idx in current_edges]
    
    return best_score, best_edges


def _local_search_edges(values, edges, all_edges, deadline, max_iters=200):
    """Local search: swap edges to reduce distinct sums/products."""
    n = len(values)
    m = len(edges)
    
    edge_attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_attrs[(u, v)] = (a + b, a * b)
    
    current_edges = [list(e) for e in edges]
    edge_set = set(tuple(e) for e in current_edges)
    
    sum_counts = Counter()
    prod_counts = Counter()
    for u, v in current_edges:
        s, p = edge_attrs[(u, v)]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    n_sums = len(sum_counts)
    n_prods = len(prod_counts)
    current_score = max(n_sums, n_prods)
    
    all_edge_list = list(all_edges)
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 1:
            break
        
        improved = False
        for i in range(m):
            if time.time() > deadline - 1:
                break
            
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
            for _ in range(20):
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
                    sum_counts[s1] -= 1
                    if sum_counts[s1] == 0:
                        del sum_counts[s1]
                    prod_counts[p1] -= 1
                    if prod_counts[p1] == 0:
                        del prod_counts[p1]
                    
                    sum_counts[s2] += 1
                    prod_counts[p2] += 1
                    
                    n_sums = new_n_sums
                    n_prods = new_n_prods
                    current_score = new_score
                    
                    current_edges[i] = list(e2)
                    edge_set.discard(e1)
                    edge_set.add(e2)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 30:
                break
    
    return current_score, current_edges


def _optimize_values(values, edges, deadline, max_iters=100):
    """Perturb values to reduce distinct sums/products."""
    n = len(values)
    current_values = list(values)
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
    
    for it in range(max_iters):
        if time.time() > deadline - 1:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 1:
                break
            
            old_val = current_values[i]
            for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20, -50, 50]:
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
        
        if not improved and it > 20:
            break
    
    return current_score, current_values


def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    # Generate candidate value sets
    candidates = _make_candidate_values()
    
    # Shuffle and limit to fit time budget
    random.shuffle(candidates)
    max_candidates = 25
    
    for ci, values in enumerate(candidates[:max_candidates]):
        if time.time() > deadline - 5:
            break
        
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        # SA edge selection
        score, edges = _sa_edge_selection(values, all_edges, deadline, num_restarts=3, steps_per_restart=30000)
        
        if edges is None or len(edges) != P['m']:
            continue
        
        # Local search polish on edges
        if time.time() < deadline - 3:
            score2, edges2 = _local_search_edges(values, edges, all_edges, deadline, max_iters=150)
            if score2 < score:
                score = score2
                edges = edges2
        
        # Optimize values
        if time.time() < deadline - 3:
            score3, values2 = _optimize_values(values, edges, deadline, max_iters=80)
            if score3 < score:
                score = score3
                values = values2
        
        # Re-optimize edges with new values
        if time.time() < deadline - 2 and score3 < score:
            score4, edges3 = _local_search_edges(values, edges, all_edges, deadline, max_iters=100)
            if score4 < score:
                score = score4
                edges = edges3
        
        if score < best_score:
            best_score = score
            best_edges = [list(e) for e in edges]
            best_values = list(values)
    
    # Fallback
    if best_edges is None:
        values1 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
        edges = list(all_edges[:P['m']])
        best_values = values1
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)
    
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
