"""
Combined structural + refinement approach.
Core idea: use multiplicative orbits (powers of 2/3) to force product collisions,
then use a fast SA with precomputed collision maps for edge selection,
followed by aggressive local search and value perturbation.
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


def _make_candidate_values():
    """Generate diverse candidate value sets optimized for product collisions."""
    candidates = []
    
    # Powers of 2 (single orbit - maximum product structure)
    pow2 = [2**i for i in range(1, 15)]  # 2, 4, 8, ..., 16384
    pow2 = [v for v in pow2 if v <= 10000]
    if len(pow2) >= 14:
        candidates.append(pow2[:14])
    
    # Mixed orbits: powers of 2 and 3
    orbit2 = [2**i for i in range(1, 15)]
    orbit3 = [3**i for i in range(1, 15)]
    orbit2 = [v for v in orbit2 if v <= 10000]
    orbit3 = [v for v in orbit3 if v <= 10000]
    
    splits = [
        (14, 0), (10, 4), (8, 6), (7, 7), (6, 8), (4, 10), (0, 14)
    ]
    for n2, n3 in splits:
        if n2 <= len(orbit2) and n3 <= len(orbit3):
            vals = orbit2[:n2] + orbit3[:n3]
            if len(set(vals)) == 14 and all(0 < v <= 10000 for v in vals):
                candidates.append(sorted(vals))
    
    # Two-prime structure: 2^a * 3^b (rich product structure)
    two_prime = []
    for a in range(0, 10):
        for b in range(0, 10):
            val = (2 ** a) * (3 ** b)
            if 0 < val <= 10000 and val not in two_prime:
                two_prime.append(val)
    two_prime.sort()
    if len(two_prime) >= 14:
        candidates.append(two_prime[:14])
        # Random samples from two_prime
        for _ in range(10):
            sampled = random.sample(two_prime, 14)
            candidates.append(sorted(sampled))
    
    # Small composites with shared factors (many product collisions)
    composites = [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108]
    if len(set(composites)) >= 14:
        candidates.append(sorted(composites[:14]))
        for _ in range(5):
            sampled = random.sample(composites, 14)
            candidates.append(sorted(sampled))
    
    # Divisors of 2520
    divs_2520 = []
    for d in range(1, 51):
        if 2520 % d == 0:
            divs_2520.append(d)
    if len(divs_2520) >= 14:
        candidates.append(sorted(divs_2520[:14]))
    
    # Divisors of 5040
    divs_5040 = []
    for d in range(1, 100):
        if 5040 % d == 0:
            divs_5040.append(d)
    if len(divs_5040) >= 14:
        candidates.append(sorted(divs_5040[:14]))
        for _ in range(3):
            sampled = random.sample(divs_5040, 14)
            candidates.append(sorted(sampled))
    
    # Even numbers (guaranteed even sums, structured products)
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
        for step in [2, 3, 4, 5, 6, 7, 8]:
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


def _fast_sa_edge_selection(values, all_edges, deadline, num_restarts=6, steps_per_restart=40000):
    """
    Fast simulated annealing with precomputed collision maps.
    Key optimization: precompute for each edge which other edges share its sum/product,
    so we can quickly estimate the impact of adding/removing an edge.
    """
    n = len(values)
    m = P['m']
    num_total = len(all_edges)
    
    # Precompute edge attributes
    edge_sums = []
    edge_prods = []
    for i, j in all_edges:
        a, b = values[i], values[j]
        edge_sums.append(a + b)
        edge_prods.append(a * b)
    
    # Build collision maps: for each sum value, list of edge indices with that sum
    sum_to_edges = {}
    prod_to_edges = {}
    for idx in range(num_total):
        s = edge_sums[idx]
        p = edge_prods[idx]
        if s not in sum_to_edges:
            sum_to_edges[s] = []
        sum_to_edges[s].append(idx)
        if p not in prod_to_edges:
            prod_to_edges[p] = []
        prod_to_edges[p].append(idx)
    
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
        
        for idx in current_edges:
            sum_counts[edge_sums[idx]] += 1
            prod_counts[edge_prods[idx]] += 1
        
        current_score = max(len(sum_counts), len(prod_counts))
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(all_edges[idx]) for idx in current_edges]
        
        # SA with adaptive temperature
        T = 0.8
        cooling = 0.9998
        min_T = 0.05
        
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
            
            # Remove e_out
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
            
            # Add e_in
            sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
            prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
            
            new_score = max(len(sum_counts), len(prod_counts))
            
            if new_score <= current_score:
                current_edges.discard(e_out)
                current_edges.add(e_in)
                current_score = new_score
            else:
                delta = new_score - current_score
                if random.random() < math.exp(-delta / T):
                    current_edges.discard(e_out)
                    current_edges.add(e_in)
                    current_score = new_score
                else:
                    # Revert
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        del sum_counts[s_in]
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        del prod_counts[p_in]
                    
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
            
            T = max(min_T, T * cooling)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(all_edges[idx]) for idx in current_edges]
    
    return best_score, best_edges


def _aggressive_local_search(values, edges, all_edges, deadline, max_rounds=300):
    """
    Aggressive local search: systematically try all edge swaps,
    with multiple passes until no improvement.
    """
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
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 1:
            break
        
        improved = False
        
        for i in range(m):
            if time.time() > deadline - 1:
                break
            
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
            # Try all possible replacements (not just random)
            for e2 in all_edge_list:
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
                    # Apply swap
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
        
        if not improved:
            break
    
    return current_score, current_edges


def _optimize_values(values, edges, deadline, max_iters=200):
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
    
    deltas = [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20, -50, 50, -100, 100, -200, 200]
    
    for it in range(max_iters):
        if time.time() > deadline - 1:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 1:
                break
            
            old_val = current_values[i]
            for delta in deltas:
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
        
        if not improved and it > 30:
            break
    
    return current_score, current_values


def _refine_pair(values, edges, all_edges, deadline):
    """
    Interleave value optimization and edge re-optimization.
    """
    current_values = list(values)
    current_edges = [list(e) for e in edges]
    current_score = _raw_score(current_values, current_edges)
    
    for phase in range(6):
        if time.time() > deadline - 2:
            break
        
        # Optimize values
        if time.time() < deadline - 3:
            vs, v2 = _optimize_values(current_values, current_edges, deadline, max_iters=50)
            if vs < current_score:
                current_score = vs
                current_values = v2
        
        # Re-optimize edges with new values
        if time.time() < deadline - 2:
            es, e2 = _aggressive_local_search(current_values, current_edges, all_edges, deadline, max_rounds=100)
            if es < current_score:
                current_score = es
                current_edges = e2
    
    return current_score, current_values, current_edges


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    # Generate candidate value sets
    candidates = _make_candidate_values()
    
    # Shuffle and limit to fit time budget
    random.shuffle(candidates)
    max_candidates = 30
    
    for ci, values in enumerate(candidates[:max_candidates]):
        if time.time() > deadline - 5:
            break
        
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        # SA edge selection
        score, edges = _fast_sa_edge_selection(values, all_edges, deadline, num_restarts=4, steps_per_restart=35000)
        
        if edges is None or len(edges) != P['m']:
            continue
        
        # Aggressive local search polish on edges
        if time.time() < deadline - 3:
            score2, edges2 = _aggressive_local_search(values, edges, all_edges, deadline, max_rounds=200)
            if score2 < score:
                score = score2
                edges = edges2
        
        # Interleaved refinement
        if time.time() < deadline - 3:
            score3, values2, edges3 = _refine_pair(values, edges, all_edges, deadline)
            if score3 < score:
                score = score3
                values = values2
                edges = edges3
        
        if score < best_score:
            best_score = score
            best_edges = [list(e) for e in edges]
            best_values = list(values)
    
    # Fallback
    if best_edges is None:
        values1 = [2, 4, 6, 8, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64]
        edges = list(all_edges[:P['m']])
        best_values = values1
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)
    
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
