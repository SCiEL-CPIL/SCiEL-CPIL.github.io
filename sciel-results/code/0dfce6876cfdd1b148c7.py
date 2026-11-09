"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85  # 5s safety margin before 90s kill


def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _score_components(values, edges):
    """Return (num_distinct_sums, num_distinct_products, sum_set, prod_set)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return len(sums), len(prods), sums, prods


def _greedy_edge_selection(values, m, n):
    """
    Greedy edge selection: iteratively pick the edge that adds the fewest
    new distinct sums and products. Ties broken by preferring edges that
    reuse existing sums/products more.
    """
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    selected = []
    sum_set = set()
    prod_set = set()
    edge_sum = {}
    edge_prod = {}
    
    # Precompute sum and product for each edge
    for e in all_edges:
        u, v = e
        edge_sum[e] = values[u] + values[v]
        edge_prod[e] = values[u] * values[v]
    
    remaining = list(all_edges)
    
    for _ in range(m):
        if time.time() > _DEADLINE:
            break
        best_edge = None
        best_score = float('inf')
        best_new_sums = 0
        best_new_prods = 0
        
        for e in remaining:
            s = edge_sum[e]
            p = edge_prod[e]
            new_s = 0 if s in sum_set else 1
            new_p = 0 if p in prod_set else 1
            # We want to minimize the increase in max(distinct_sums, distinct_prods)
            cur_max = max(len(sum_set), len(prod_set))
            new_max = max(len(sum_set) + new_s, len(prod_set) + new_p)
            increase = new_max - cur_max
            # Secondary: minimize total new distinct values
            total_new = new_s + new_p
            # Tertiary: prefer edges with smaller values (more collision potential)
            val_key = values[e[0]] + values[e[1]]
            
            score_tuple = (increase, total_new, val_key)
            if score_tuple < best_score if isinstance(best_score, tuple) else True:
                if best_edge is None or score_tuple < best_score:
                    best_score = score_tuple
                    best_edge = e
                    best_new_sums = new_s
                    best_new_prods = new_p
        
        if best_edge is None:
            break
        
        selected.append(best_edge)
        sum_set.add(edge_sum[best_edge])
        prod_set.add(edge_prod[best_edge])
        remaining.remove(best_edge)
    
    return selected


def _hill_climb_edges(values, edges, m, n, max_iter=30000):
    """
    Hill climbing on edge set: try swapping one edge for another to reduce score.
    Uses incremental score updates where possible.
    """
    all_edges = list(combinations(range(n), 2))
    edge_set = set(edges)
    
    # Precompute
    edge_sum = {}
    edge_prod = {}
    for e in all_edges:
        u, v = e
        edge_sum[e] = values[u] + values[v]
        edge_prod[e] = values[u] * values[v]
    
    # Current state
    sum_counts = {}
    prod_counts = {}
    for e in edge_set:
        s = edge_sum[e]
        p = edge_prod[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    def _get_score():
        return max(len(sum_counts), len(prod_counts))
    
    current_score = _get_score()
    best_score = current_score
    best_edge_set = set(edge_set)
    
    # Build list of non-selected edges
    non_selected = [e for e in all_edges if e not in edge_set]
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if current_score <= 3:  # Can't do much better
            break
        
        # Pick a random edge to remove
        remove_edge = random.choice(list(edge_set))
        
        # Try a few candidate edges to add
        tried = 0
        max_tries = 5
        while tried < max_tries:
            if time.time() > _DEADLINE:
                break
            add_edge = random.choice(non_selected)
            tried += 1
            
            # Compute new score if we swap
            s_rem = edge_sum[remove_edge]
            p_rem = edge_prod[remove_edge]
            s_add = edge_sum[add_edge]
            p_add = edge_prod[add_edge]
            
            # Count changes
            new_sum_count = len(sum_counts)
            new_prod_count = len(prod_counts)
            
            # Remove remove_edge
            sum_counts[s_rem] -= 1
            if sum_counts[s_rem] == 0:
                del sum_counts[s_rem]
                new_sum_count -= 1
            prod_counts[p_rem] -= 1
            if prod_counts[p_rem] == 0:
                del prod_counts[p_rem]
                new_prod_count -= 1
            
            # Add add_edge
            if s_add not in sum_counts:
                sum_counts[s_add] = 1
                new_sum_count += 1
            else:
                sum_counts[s_add] += 1
            if p_add not in prod_counts:
                prod_counts[p_add] = 1
                new_prod_count += 1
            else:
                prod_counts[p_add] += 1
            
            new_score = max(new_sum_count, new_prod_count)
            
            if new_score <= current_score:
                # Accept the swap
                edge_set.discard(remove_edge)
                edge_set.add(add_edge)
                # Update non_selected
                non_selected.remove(add_edge)
                non_selected.append(remove_edge)
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edge_set = set(edge_set)
                break
            else:
                # Revert
                if sum_counts.get(s_add, 0) == 1:
                    del sum_counts[s_add]
                else:
                    sum_counts[s_add] -= 1
                if prod_counts.get(p_add, 0) == 1:
                    del prod_counts[p_add]
                else:
                    prod_counts[p_add] -= 1
                sum_counts[s_rem] = sum_counts.get(s_rem, 0) + 1
                prod_counts[p_rem] = prod_counts.get(p_rem, 0) + 1
                # Revert new_sum_count/new_prod_count isn't needed since we reverted
    
    return best_score, list(best_edge_set)


def _try_value_set(values, m, n):
    """Try a value set: greedy init + hill climb."""
    if time.time() > _DEADLINE:
        return None
    
    # Greedy initialization
    edges = _greedy_edge_selection(values, m, n)
    if len(edges) < m:
        # Pad with random edges
        all_edges = list(combinations(range(n), 2))
        edge_set = set(edges)
        for e in all_edges:
            if len(edge_set) >= m:
                break
            if e not in edge_set:
                edge_set.add(e)
        edges = list(edge_set)
    
    # Hill climb
    score, final_edges = _hill_climb_edges(values, edges, m, n, max_iter=20000)
    
    return score, values, final_edges


def _generate_value_candidates(n, max_val):
    """Generate diverse candidate value sets."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Small even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Highly composite-like
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # 4. Small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    candidates.append(primes[:n])
    
    # 5. Arithmetic progressions
    for start in range(1, 8):
        for step in range(1, 5):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and min(vals) >= 1:
                candidates.append(vals)
    
    # 6. Geometric-like (small)
    for base in range(2, 5):
        vals = []
        v = 1
        for _ in range(n):
            vals.append(v)
            v *= base
            if v > max_val:
                break
        if len(vals) == n and len(set(vals)) == n:
            candidates.append(vals)
    
    # 7. Random small values
    random.seed(42)
    for _ in range(30):
        vals = sorted(random.sample(range(1, 40), n))
        candidates.append(vals)
    
    # 8. Clustered values (many close together for collision)
    for base in range(1, 15):
        vals = []
        for i in range(n):
            v = base + random.randint(0, 4)
            vals.append(v)
        vals = sorted(set(vals))
        while len(vals) < n:
            vals.append(base + random.randint(0, 8))
        vals = sorted(vals[:n])
        if len(set(vals)) == n and max(vals) <= max_val:
            candidates.append(vals)
    
    # 9. Values with many shared factors (for product collisions)
    # Use multiples of small numbers
    for d in [2, 3, 4, 5, 6]:
        vals = []
        for i in range(n):
            vals.append(d * (i + 1))
        if max(vals) <= max_val and len(set(vals)) == n:
            candidates.append(vals)
    
    # 10. Mix of very small and some larger
    random.seed(123)
    for _ in range(15):
        small = sorted(random.sample(range(1, 10), 7))
        large = sorted(random.sample(range(10, 30), 7))
        vals = sorted(small + large)
        if len(set(vals)) == n and max(vals) <= max_val:
            candidates.append(vals)
    
    return candidates


def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m  # worst case
    best_result = None
    
    candidates = _generate_value_candidates(n, max_val)
    
    # Shuffle candidates for diversity
    random.seed(7)
    random.shuffle(candidates)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        result = _try_value_set(vals, m, n)
        if result is None:
            continue
        score, values, edges = result
        if score < best_score:
            best_score = score
            best_result = (values, edges)
            if best_score <= 3:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
