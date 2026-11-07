"""
Structural shift: Model the problem as a constraint-satisfaction search over value sets
and edge sets simultaneously, using a novel approach based on:
1. Systematic exploration of value structures with high product/sum collision potential
2. A priority-queue based greedy edge selection that maximizes collisions
3. Aggressive joint optimization of values and edges

The key insight is to treat edge selection as selecting 35 edges from 91 possible edges
to maximize the number of sum/product collisions, which is equivalent to minimizing
distinct sums and products. We use a greedy approach with a priority queue where
edges are ranked by their marginal contribution to collision count.
"""
import time
import random
import math
import heapq
from collections import Counter, defaultdict
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


def _make_structural_values():
    """
    Generate value sets with strong structural properties for collision.
    Focus on:
    - Powers of small primes (2, 3, 5) for product structure
    - Divisors of highly composite numbers
    - Values with many common factors
    """
    candidates = []
    
    # Strategy 1: Powers of 2 only (maximum product structure)
    pow2 = [2**i for i in range(1, 15) if 2**i <= 10000]
    if len(pow2) >= 14:
        candidates.append(pow2[:14])
    
    # Strategy 2: Powers of 2 and 3 mixed
    orbit2 = [2**i for i in range(1, 15) if 2**i <= 10000]
    orbit3 = [3**i for i in range(1, 15) if 3**i <= 10000]
    for n2 in range(0, 15):
        n3 = 14 - n2
        if n2 <= len(orbit2) and n3 <= len(orbit3):
            vals = sorted(orbit2[:n2] + orbit3[:n3])
            if len(set(vals)) == 14:
                candidates.append(vals)
    
    # Strategy 3: 2^a * 3^b grid (rich multiplicative structure)
    two_three = []
    for a in range(0, 10):
        for b in range(0, 10):
            val = (2 ** a) * (3 ** b)
            if 0 < val <= 10000:
                two_three.append(val)
    two_three = sorted(set(two_three))
    if len(two_three) >= 14:
        candidates.append(two_three[:14])
        # Random subsets
        for _ in range(15):
            sampled = random.sample(two_three, 14)
            candidates.append(sorted(sampled))
    
    # Strategy 4: Divisors of highly composite numbers
    for hc in [2520, 5040, 7560, 10080, 15120, 20160, 27720, 45360, 50400, 83160]:
        divs = []
        for d in range(1, int(math.sqrt(hc)) + 1):
            if hc % d == 0:
                divs.append(d)
                if d != hc // d:
                    divs.append(hc // d)
        divs = sorted(set(divs))
        divs = [d for d in divs if d <= 10000]
        if len(divs) >= 14:
            candidates.append(divs[:14])
            for _ in range(3):
                sampled = random.sample(divs, 14)
                candidates.append(sorted(sampled))
    
    # Strategy 5: Small composites with shared factors
    composites_pool = []
    for i in range(2, 100):
        for j in range(2, 100):
            v = i * j
            if 0 < v <= 10000:
                composites_pool.append(v)
    composites_pool = sorted(set(composites_pool))
    if len(composites_pool) >= 14:
        candidates.append(composites_pool[:14])
        for _ in range(10):
            sampled = random.sample(composites_pool[:200], 14)
            candidates.append(sorted(sampled))
    
    # Strategy 6: Multiples of small numbers
    for base in [2, 3, 4, 5, 6, 8, 10, 12]:
        mults = [base * i for i in range(1, 100) if base * i <= 10000]
        if len(mults) >= 14:
            candidates.append(mults[:14])
            for _ in range(3):
                sampled = random.sample(mults[:50], 14)
                candidates.append(sorted(sampled))
    
    # Strategy 7: Arithmetic progressions with small step
    for start in range(1, 10):
        for step in range(1, 10):
            vals = [start + i * step for i in range(14)]
            if all(0 < v <= 10000 for v in vals):
                candidates.append(vals)
    
    # Strategy 8: Values from a small set of primes' products
    small_primes = [2, 3, 5, 7]
    prime_products = []
    for subset_size in range(1, 5):
        for combo in combinations(small_primes, subset_size):
            prod = 1
            for p in combo:
                prod *= p
            if prod <= 10000:
                prime_products.append(prod)
            # Also multiply by small powers
            for power in range(1, 5):
                val = prod * (2 ** power)
                if val <= 10000:
                    prime_products.append(val)
    prime_products = sorted(set(prime_products))
    if len(prime_products) >= 14:
        candidates.append(prime_products[:14])
        for _ in range(10):
            sampled = random.sample(prime_products[:100], 14)
            candidates.append(sorted(sampled))
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen and len(set(c)) == 14 and all(0 < v <= 1000 for v in c) or all(0 < v <= 10000 for v in c):
            seen.add(key)
            unique_candidates.append(list(c))
    
    return unique_candidates


def _greedy_edge_selection_with_pq(values, all_edges, deadline):
    """
    Priority-queue based greedy edge selection.
    
    Start with all 91 possible edges. Iteratively select edges that maximize
    collision count (i.e., edges whose sum/product already appears in selected edges).
    
    This is a structural shift from sequential addition to a global ranking approach.
    """
    n = len(values)
    m = P['m']
    
    # Precompute all edge attributes
    edge_attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_attrs[(u, v)] = (a + b, a * b)
    
    # Build collision maps
    sum_to_edges = defaultdict(list)
    prod_to_edges = defaultdict(list)
    for e, (s, p) in edge_attrs.items():
        sum_to_edges[s].append(e)
        prod_to_edges[p].append(e)
    
    # Score each edge by its potential for collisions
    # An edge is "good" if its sum or product matches many other edges
    edge_score = {}
    for e, (s, p) in edge_attrs.items():
        sum_freq = len(sum_to_edges[s])
        prod_freq = len(prod_to_edges[p])
        edge_score[e] = sum_freq + prod_freq
    
    # Use a max-heap (negative scores for min-heap)
    heap = []
    for e in all_edges:
        heapq.heappush(heap, (-edge_score[e], e))
    
    selected = []
    selected_set = set()
    sum_counts = Counter()
    prod_counts = Counter()
    
    current_time = time.time()
    
    while len(selected) < m and heap:
        if time.time() > deadline - 2:
            break
        
        neg_score, e = heapq.heappop(heap)
        
        if e in selected_set:
            continue
        
        s, p = edge_attrs[e]
        
        # Calculate marginal benefit: how many new collisions this edge creates
        # If sum s already appears k times, adding this edge creates k new sum collisions
        # If product p already appears l times, adding this edge creates l new product collisions
        sum_collisions = sum_counts.get(s, 0)
        prod_collisions = prod_counts.get(p, 0)
        
        # Also consider: if this edge's sum/product is new, it adds 1 to distinct count
        # We want to maximize collisions, so prefer edges with high sum_collisions + prod_collisions
        marginal = sum_collisions + prod_collisions
        
        # Accept the edge
        selected.append(e)
        selected_set.add(e)
        sum_counts[s] += 1
        prod_counts[p] += 1
        
        # Update scores of remaining edges (approximate: just re-push with updated info)
        # This is expensive, so we do a simplified update
        # Actually, for efficiency, we just continue popping from the heap
        # The heap scores are static, but the marginal benefit changes as we add edges
    
    return selected[:m]


def _local_search_edges(values, edges, all_edges, deadline, max_rounds=200):
    """
    Local search: try swapping each edge with any other edge to improve score.
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
        if time.time() > deadline - 2:
            break
        
        improved = False
        
        for i in range(m):
            if time.time() > deadline - 2:
                break
            
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
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


def _joint_optimize(values, edges, all_edges, deadline, max_iters=100):
    """
    Joint optimization: alternate between value perturbation and edge re-optimization.
    """
    current_values = list(values)
    current_edges = [list(e) for e in edges]
    current_score = _raw_score(current_values, current_edges)
    
    edge_pairs = [(u, v) for u, v in current_edges]
    
    def _score(vals):
        sums = set()
        prods = set()
        for u, v in edge_pairs:
            a, b = vals[u], vals[v]
            sums.add(a + b)
            prods.add(a * b)
        return max(len(sums), len(prods))
    
    deltas = [-2, -1, 1, 2, -5, 5, -10, 10, -20, 20, -50, 50]
    
    for it in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        improved = False
        
        # Try value perturbation
        for i in range(len(current_values)):
            if time.time() > deadline - 3:
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
        
        if not improved:
            # Try edge re-optimization
            es, e2 = _local_search_edges(current_values, current_edges, all_edges, deadline, max_rounds=50)
            if es < current_score:
                current_score = es
                current_edges = e2
                edge_pairs = [(u, v) for u, v in current_edges]
                improved = True
        
        if not improved and it > 20:
            break
    
    return current_score, current_values, current_edges


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    # Generate candidate value sets
    candidates = _make_structural_values()
    
    # Shuffle and limit
    random.shuffle(candidates)
    max_candidates = 40
    
    for ci, values in enumerate(candidates[:max_candidates]):
        if time.time() > deadline - 5:
            break
        
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        # Greedy edge selection with priority queue
        edges = _greedy_edge_selection_with_pq(values, all_edges, deadline)
        
        if len(edges) != P['m']:
            # Fallback: take first m edges
            edges = [list(e) for e in all_edges[:P['m']]]
        
        # Local search polish
        if time.time() < deadline - 3:
            score, edges2 = _local_search_edges(values, edges, all_edges, deadline, max_rounds=150)
            if score < _raw_score(values, edges):
                score = score
                edges = edges2
            else:
                score = _raw_score(values, edges)
        
        # Joint optimization
        if time.time() < deadline - 3:
            score3, values2, edges3 = _joint_optimize(values, edges, all_edges, deadline, max_iters=80)
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
        edges = [list(e) for e in all_edges[:P['m']]]
        best_values = values1
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)
    
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
