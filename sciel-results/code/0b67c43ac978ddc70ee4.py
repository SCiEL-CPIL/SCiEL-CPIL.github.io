"""Model-and-optimize: parameterize 14 values and 35 edges; fit against max(distinct_sums, distinct_products)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import Counter, defaultdict

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    """Compute the raw score: max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    """Generate all C(n,2) possible edges."""
    return list(itertools.combinations(range(n), 2))

def _make_powers_of_prime_values(base, max_exp=14):
    """Generate powers of a prime base, all <= 10000."""
    vals = []
    val = 1
    for _ in range(max_exp):
        if val > 10000:
            break
        vals.append(val)
        val *= base
    # Ensure we have exactly 14 distinct values
    while len(vals) < 14:
        # Extend with multiples
        for i in range(len(vals)):
            candidate = vals[i] * base
            if candidate <= 10000 and candidate not in vals:
                vals.append(candidate)
                break
        else:
            # Fallback: add small distinct numbers
            for x in range(1, 10001):
                if x not in vals:
                    vals.append(x)
                    break
    return sorted(vals[:14])

def _make_two_prime_structure():
    """Values of form 2^a * 3^b with structured exponents for collision."""
    candidates = []
    for a in range(0, 8):
        for b in range(0, 8):
            val = (2 ** a) * (3 ** b)
            if 0 < val <= 10000 and val not in candidates:
                candidates.append(val)
    candidates.sort()
    # Select 14 that maximize product collision potential
    return candidates[:14]

def _make_optimal_structure():
    """
    Use values where products collide heavily.
    Key insight: if values are all powers of 2, then v_i * v_j = 2^(e_i + e_j).
    Products collide when e_i + e_j is the same.
    Sums: 2^a + 2^b. These collide less easily, but we can choose exponents wisely.
    
    Alternative: use values that are all divisors of a highly composite number.
    """
    # Strategy: use 14 divisors of 2520 = 2^3 * 3^2 * 5 * 7
    # This gives many product collisions
    base = 2520
    divisors = []
    for d in range(1, int(base**0.5) + 1):
        if base % d == 0:
            divisors.append(d)
            if d != base // d:
                divisors.append(base // d)
    divisors.sort()
    if len(divisors) >= 14:
        return divisors[:14]
    
    # Fallback: powers of 2
    return _make_powers_of_prime_values(2, 14)

def _score_edge_set(values, edges):
    """Score a set of edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _group_edges_by_sum_product(values, all_edges):
    """
    Group edges by their (sum, product) pair.
    Returns dict: (sum, product) -> list of edges
    """
    groups = defaultdict(list)
    for u, v in all_edges:
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        groups[(s, p)].append((u, v))
    return groups

def _greedy_with_collision(values, target_m, all_edges):
    """
    Greedy selection prioritizing edges that reuse existing sums AND products.
    Key: maximize (sum_reuse + prod_reuse), then minimize new distinct values.
    """
    n = len(values)
    edges_chosen = []
    chosen_set = set()
    sum_counts = Counter()
    prod_counts = Counter()
    
    # Precompute all edge attributes
    edge_info = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        edge_info.append((u, v, s, p))
    
    for _ in range(target_m):
        best_edge = None
        best_key = None
        for (u, v, s, p) in edge_info:
            if (u, v) in chosen_set:
                continue
            s_count = sum_counts.get(s, 0)
            p_count = prod_counts.get(p, 0)
            # Primary: maximize reuse. Secondary: prefer smaller values (may help future collisions)
            key = (s_count + p_count, -s, -p)
            if best_key is None or key > best_key:
                best_key = key
                best_edge = (u, v, s, p)
        
        if best_edge is None:
            break
        
        u, v, s, p = best_edge
        edges_chosen.append((u, v))
        chosen_set.add((u, v))
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    return [list(e[:2]) for e in edges_chosen]

def _local_search_swap(values, edges, all_edges, deadline, max_iters=100):
    """
    Local search: try swapping edges to improve score.
    Optimized with incremental counting.
    """
    n = len(values)
    m = len(edges)
    
    # Precompute edge attributes
    edge_attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_attrs[(u, v)] = (a + b, a * b)
    
    current_edges = [list(e) for e in edges]
    edge_set = set(tuple(e) for e in current_edges)
    
    # Current score state
    sum_counts = Counter()
    prod_counts = Counter()
    for u, v in current_edges:
        s, p = edge_attrs[(u, v)]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    def _current_score():
        n_sums = sum(1 for c in sum_counts.values() if c > 0)
        n_prods = sum(1 for c in prod_counts.values() if c > 0)
        return max(n_sums, n_prods)
    
    current_score = _current_score()
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try all edge swaps
        for i in range(m):
            if time.time() > deadline:
                break
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
            for e2 in all_edges:
                if e2 == e1 or e2 in edge_set:
                    continue
                
                s2, p2 = edge_attrs[e2]
                
                # Simulate swap: remove e1, add e2
                old_s_count = sum_counts[s1]
                old_p_count = prod_counts[p1]
                new_s_count = sum_counts.get(s2, 0)
                new_p_count = prod_counts.get(p2, 0)
                
                # After removal
                sum_counts[s1] = old_s_count - 1
                prod_counts[p1] = old_p_count - 1
                if sum_counts[s1] == 0:
                    del sum_counts[s1]
                if prod_counts[p1] == 0:
                    del prod_counts[p1]
                
                # After addition
                sum_counts[s2] = new_s_count + 1
                prod_counts[p2] = new_p_count + 1
                
                new_score = _current_score()
                
                if new_score < current_score:
                    # Accept the swap
                    current_edges[i] = list(e2)
                    edge_set.remove(e1)
                    edge_set.add(e2)
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    sum_counts[s1] = old_s_count
                    prod_counts[p1] = old_p_count
                    sum_counts[s2] = new_s_count
                    if new_s_count == 0:
                        del sum_counts[s2]
                    prod_counts[p2] = new_p_count
                    if new_p_count == 0:
                        del prod_counts[p2]
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_score, current_edges

def _optimize_for_values(values, all_edges, deadline):
    """
    Optimize edge selection for given values.
    """
    n = len(values)
    m = P['m']
    
    best_score = float('inf')
    best_edges = None
    
    # Greedy + local search
    edges = _greedy_with_collision(values, m, all_edges)
    if len(edges) != m:
        return best_score, None
    
    score, edges = _local_search_swap(values, edges, all_edges, deadline, max_iters=80)
    best_score = min(best_score, score)
    best_edges = [list(e) for e in edges]
    
    return best_score, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    # Strategy 1: Powers of 2 (max product collision)
    if time.time() < deadline - 10:
        values = _make_powers_of_prime_values(2, 14)
        if len(set(values)) == 14 and all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            score, edges = _optimize_for_values(values, all_edges, deadline)
            if edges is not None and score < best_score:
                best_score = score
                best_edges = edges
                best_values = values[:]
    
    # Strategy 2: Divisors of 2520
    if time.time() < deadline - 10:
        values = _make_optimal_structure()
        if len(set(values)) == 14 and all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            score, edges = _optimize_for_values(values, all_edges, deadline)
            if edges is not None and score < best_score:
                best_score = score
                best_edges = edges
                best_values = values[:]
    
    # Strategy 3: Two-prime structure
    if time.time() < deadline - 10:
        values = _make_two_prime_structure()
        if len(set(values)) == 14 and all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            score, edges = _optimize_for_values(values, all_edges, deadline)
            if edges is not None and score < best_score:
                best_score = score
                best_edges = edges
                best_values = values[:]
    
    # Strategy 4: Small composites with shared factors
    if time.time() < deadline - 10:
        values = [2, 4, 6, 8, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96]
        if len(set(values)) == 14 and all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            score, edges = _optimize_for_values(values, all_edges, deadline)
            if edges is not None and score < best_score:
                best_score = score
                best_edges = edges
                best_values = values[:]
    
    # Strategy 5: Geometric with ratio 2
    if time.time() < deadline - 10:
        values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        if len(set(values)) == 14 and all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            score, edges = _optimize_for_values(values, all_edges, deadline)
            if edges is not None and score < best_score:
                best_score = score
                best_edges = edges
                best_values = values[:]
    
    # Fallback
    if best_edges is None:
        values = _make_powers_of_prime_values(2, 14)
        edges = _greedy_with_collision(values, P['m'], all_edges)
        best_values = values
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)
    
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
