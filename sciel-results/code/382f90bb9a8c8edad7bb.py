"""
CSP-style approach: partition values into multiplicative orbits to force product collisions,
then use backtracking with forward checking to select 35 edges that minimize
max(#distinct sums, #distinct products).
"""
import time
import itertools
import random
from collections import Counter, defaultdict

DEADLINE = 85.0

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    return list(itertools.combinations(range(n), 2))

def _generate_orbit_values(n=14, max_val=10000):
    """
    Generate value sets where many values share multiplicative structure,
    forcing product collisions.
    
    Strategy: pick a base set of "orbit centers" and generate values as
    center * k where k comes from a small set of multipliers.
    This way, products of values from the same orbit tend to collide.
    
    Also try: values that are all divisors of a highly composite number,
    so products of pairs are also divisors (bounded distinct products).
    """
    value_sets = []
    
    # Strategy 1: Divisors of a highly composite number
    # 27720 = 2^3 * 3^2 * 5 * 7 * 11
    # 75600 = 2^3 * 3^3 * 5^2 * 7
    # 2520 = 2^3 * 3^2 * 5 * 7
    for base in [2520, 7560, 27720, 5040, 1260, 1680, 420, 840, 630, 210]:
        divs = []
        for i in range(1, base + 1):
            if base % i == 0:
                divs.append(i)
        if len(divs) >= n:
            # Take n distinct divisors
            # Prefer divisors that are "spread out" to create sum diversity
            # but keep them small enough that products stay bounded
            random.shuffle(divs)
            # Actually, let's be smarter: take divisors in a structured way
            divs_sorted = sorted(divs)
            # Take a mix: some small, some large
            selected = []
            # First, take the smallest n//2 and largest n//2
            half = n // 2
            selected.extend(divs_sorted[:half])
            selected.extend(divs_sorted[-(n - half):])
            # Ensure distinct
            if len(set(selected)) == n:
                value_sets.append(selected)
    
    # Strategy 2: Values = a_i * m where a_i are small distinct integers and m is a common factor
    # This makes products = a_i * a_j * m^2, so distinct products = distinct (a_i * a_j)
    for m in [2, 3, 4, 5, 6, 7, 10, 12]:
        for base_set in [[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                         [1,2,3,4,5,6,7,8,9,10,11,12,15,16],
                         [1,2,3,4,5,6,7,8,9,10,12,14,16,18]]:
            vals = [x * m for x in base_set]
            if len(set(vals)) == n and all(1 <= v <= max_val for v in vals):
                value_sets.append(vals)
    
    # Strategy 3: Powers of 2 times small odd numbers
    # 2^a * b where b is odd
    # Products: 2^(a1+a2) * b1*b2
    for pattern in [
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        [1,2,3,4,5,6,7,8,9,10,11,12,15,16],
        [1,2,3,4,5,6,7,8,9,10,12,14,16,20],
    ]:
        vals = [p * 8 for p in pattern]
        if len(set(vals)) == n and all(1 <= v <= max_val for v in vals):
            value_sets.append(vals)
    
    # Strategy 4: Additive collision sets (many pairs sum to same value)
    # Combined with multiplicative structure
    for target_sum in [30, 40, 50, 60, 80, 100]:
        pairs = []
        used = set()
        for i in range(1, target_sum // 2 + 1):
            j = target_sum - i
            if i != j and i not in used and j not in used:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
        # Take n values from these pairs
        vals = []
        for a, b in pairs:
            if len(vals) < n:
                vals.append(a)
                if len(vals) < n:
                    vals.append(b)
        while len(vals) < n:
            x = random.randint(1, max_val)
            if x not in vals:
                vals.append(x)
        if len(set(vals)) == n:
            value_sets.append(vals[:n])
    
    # Strategy 5: Random but structured
    random.seed(123)
    for _ in range(10):
        # Generate values with some multiplicative structure
        base = random.randint(1, 100)
        multipliers = random.sample(range(1, 20), n)
        vals = [base * m for m in multipliers]
        if len(set(vals)) == n and all(1 <= v <= max_val for v in vals):
            value_sets.append(vals)
    
    # Deduplicate
    unique_sets = []
    seen = set()
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(vs)
    
    return unique_sets

def _backtrack_edge_selection(values, target_m, time_limit):
    """
    Backtracking search to select target_m edges minimizing max(distinct_sums, distinct_products).
    Uses forward checking: at each step, only consider edges that don't immediately
    make it impossible to achieve a good score.
    """
    n = len(values)
    all_e = _all_edges(n)
    start_time = time.time()
    
    # Precompute sums and products
    edge_data = []
    for u, v in all_e:
        a, b = values[u], values[v]
        edge_data.append((u, v, a + b, a * b))
    
    best_score = float('inf')
    best_edges = None
    
    # Sort edges by "collision potential" - edges whose sum/product appear most frequently
    sum_freq = Counter(s for _, _, s, _ in edge_data)
    prod_freq = Counter(p for _, _, _, p in edge_data)
    
    def edge_priority(idx):
        u, v, s, p = edge_data[idx]
        return (sum_freq[s], prod_freq[p])
    
    sorted_indices = sorted(range(len(all_e)), key=lambda i: edge_priority(i), reverse=True)
    
    # Backtracking with pruning
    def backtrack(edge_idx_list, current_edges, sums, prods, depth):
        nonlocal best_score, best_edges
        
        if time.time() - start_time > time_limit:
            return
        
        if len(current_edges) == target_m:
            score = max(len(sums), len(prods))
            if score < best_score:
                best_score = score
                best_edges = list(current_edges)
            return
        
        if len(current_edges) + (len(edge_idx_list) - depth) < target_m:
            return
        
        # Pruning: if current max already exceeds best, skip
        current_max = max(len(sums), len(prods))
        if current_max >= best_score and len(current_edges) > target_m - 5:
            return
        
        for i in range(depth, len(edge_idx_list)):
            if time.time() - start_time > time_limit:
                return
            
            idx = edge_idx_list[i]
            u, v, s, p = edge_data[idx]
            
            # Check if this edge is already in current_edges
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            
            new_sums = sums | {s}
            new_prods = prods | {p}
            
            # Forward checking: if adding this edge makes it impossible
            # to stay under best_score, skip
            new_max = max(len(new_sums), len(new_prods))
            if new_max >= best_score:
                continue
            
            current_edges.add((u, v))
            backtrack(edge_idx_list, current_edges, new_sums, new_prods, i + 1)
            current_edges.discard((u, v))
            
            if best_score <= 5:  # Very good solution found
                return
    
    # Use a subset of edges to make backtracking feasible
    # Take top K edges by collision potential
    K = min(60, len(sorted_indices))
    top_edges = sorted_indices[:K]
    
    backtrack(top_edges, set(), set(), set(), 0)
    
    return best_edges, best_score

def _greedy_with_lookahead(values, target_m, time_limit):
    """
    Greedy edge selection with 2-step lookahead.
    At each step, consider each candidate edge, and for each, consider
    the best second edge to add, then pick the pair that minimizes
    the resulting max distinct count.
    """
    n = len(values)
    all_e = _all_edges(n)
    start_time = time.time()
    
    edge_data = {}
    for u, v in all_e:
        a, b = values[u], values[v]
        edge_data[(u, v)] = (a + b, a * b)
    
    sums = set()
    prods = set()
    edges = []
    used_vertices_degree = [0] * n
    
    remaining = list(all_e)
    
    for step in range(target_m):
        if time.time() - start_time > time_limit:
            break
        
        best_score = float('inf')
        best_edge = None
        best_sums = None
        best_prods = None
        
        for e in remaining:
            u, v = e
            s, p = edge_data[e]
            
            new_sums = sums | {s}
            new_prods = prods | {p}
            
            # 1-step score
            score1 = max(len(new_sums), len(new_prods))
            
            if score1 >= best_score:
                continue
            
            # 2-step lookahead: for each candidate second edge, compute score
            best_second_score = float('inf')
            for e2 in remaining:
                if e2 == e:
                    continue
                u2, v2 = e2
                s2, p2 = edge_data[e2]
                
                # Check if e2 shares a vertex with e (allowed, but might be less ideal)
                new_sums2 = new_sums | {s2}
                new_prods2 = new_prods | {p2}
                score2 = max(len(new_sums2), len(new_prods2))
                
                if score2 < best_second_score:
                    best_second_score = score2
            
            # Combine: prefer edges where even the best second edge gives good score
            combined = max(score1, best_second_score)
            
            if combined < best_score:
                best_score = combined
                best_edge = e
                best_sums = new_sums
                best_prods = new_prods
        
        if best_edge is None:
            # Fallback: pick edge with minimum 1-step score
            for e in remaining:
                u, v = e
                s, p = edge_data[e]
                new_sums = sums | {s}
                new_prods = prods | {p}
                score1 = max(len(new_sums), len(new_prods))
                if score1 < best_score:
                    best_score = score1
                    best_edge = e
                    best_sums = new_sums
                    best_prods = new_prods
        
        if best_edge is None:
            break
        
        u, v = best_edge
        s, p = edge_data[best_edge]
        sums = best_sums
        prods = best_prods
        edges.append([u, v])
        remaining.remove(best_edge)
    
    return edges, _raw_score(values, edges)

def _local_search(values, edges, max_iters, timeout):
    """
    Local search: swap edges to improve score.
    Uses simulated annealing-like acceptance.
    """
    n = len(values)
    all_e_set = set(_all_edges(n))
    current_edges = set(tuple(sorted(e)) for e in edges)
    current_score = _raw_score(values, list(current_edges))
    
    edge_props = {}
    for u, v in all_e_set:
        a, b = values[u], values[v]
        edge_props[(u, v)] = (a + b, a * b)
    
    start_time = time.time()
    best_score = current_score
    best_edges = set(current_edges)
    
    # Precompute for faster evaluation
    def compute_score(edge_set):
        sums = set()
        prods = set()
        for u, v in edge_set:
            s, p = edge_props[(u, v)]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))
    
    for _ in range(max_iters):
        if time.time() > timeout:
            break
        
        # Pick random edge to remove
        e_remove = random.choice(list(current_edges))
        # Pick random edge to add
        candidates = list(all_e_set - current_edges)
        if not candidates:
            break
        e_add = random.choice(candidates)
        
        new_edges = current_edges - {e_remove} | {e_add}
        new_score = compute_score(new_edges)
        
        if new_score <= current_score:
            current_edges = new_edges
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        else:
            # Accept with some probability (annealing-like)
            if random.random() < 0.1:
                current_edges = new_edges
                current_score = new_score
    
    return list(best_edges), best_score

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    n = 14
    m = 35
    max_val = 10000
    
    # Generate candidate value sets
    value_sets = _generate_orbit_values(n, max_val)
    
    # Also add some hand-crafted sets
    value_sets.append(list(range(1, n + 1)))
    value_sets.append([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Deduplicate
    unique_sets = []
    seen = set()
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen and len(set(vs)) == n:
            seen.add(key)
            unique_sets.append(vs)
    
    # Time budget per value set
    time_per_set = 3.0
    
    for vs in unique_sets:
        if time.time() > DEADLINE:
            break
        
        set_start = time.time()
        
        # Try greedy with lookahead
        edges_g, score_g = _greedy_with_lookahead(vs, m, set_start + time_per_set * 0.4)
        if score_g < best_score:
            best_score = score_g
            best_result = {'values': vs, 'edges': edges_g}
        
        # Try backtracking if time permits
        if time.time() < DEADLINE - 5.0:
            edges_b, score_b = _backtrack_edge_selection(vs, m, set_start + time_per_set * 0.7)
            if edges_b is not None and score_b < best_score:
                best_score = score_b
                best_result = {'values': vs, 'edges': edges_b}
        
        # Local search polish
        if time.time() < DEADLINE - 2.0:
            base_edges = best_result['edges'] if best_result['values'] == vs else edges_g
            edges_ls, score_ls = _local_search(vs, base_edges, 20000, time.time() + 1.5)
            if score_ls < best_score:
                best_score = score_ls
                best_result = {'values': vs, 'edges': edges_ls}
    
    # Final local search on best result with more time
    if best_result and time.time() < DEADLINE - 1.0:
        edges_final, score_final = _local_search(
            best_result['values'], best_result['edges'], 50000, DEADLINE - 1.0
        )
        if score_final < best_score:
            best_score = score_final
            best_result = {'values': best_result['values'], 'edges': edges_final}
    
    # Fallback
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(itertools.combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}
    
    # Ensure types
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
