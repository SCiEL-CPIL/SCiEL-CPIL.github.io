"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict

_DEADLINE = time.time() + 82  # safety margin before 90s kill

def _score_fast(values, edges):
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
    """Return (num_distinct_sums, num_distinct_products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return len(sums), len(prods)

def _greedy_edge_selection(values, m, n):
    """
    Greedy edge selection: iteratively pick the edge that adds the fewest
    new distinct sums and products.
    """
    current_sums = set()
    current_prods = set()
    selected_edges = []
    
    # Precompute all possible edges with their sum and product
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
    
    for _ in range(m):
        if time.time() > _DEADLINE:
            break
        best_edge = None
        best_new_sums = None
        best_new_prods = None
        best_score = float('inf')
        
        for (i, j, s, p) in all_edges:
            if (i, j) in [(e[0], e[1]) for e in selected_edges]:
                continue
            
            new_sums = current_sums | {s}
            new_prods = current_prods | {p}
            score = max(len(new_sums), len(new_prods))
            
            if score < best_score:
                best_score = score
                best_edge = (i, j)
                best_new_sums = new_sums
                best_new_prods = new_prods
                if best_score <= 2:
                    break
        
        if best_edge is None:
            break
        
        selected_edges.append(best_edge)
        current_sums = best_new_sums
        current_prods = best_new_prods
    
    return selected_edges

def _local_search_edges(values, edges, n, m, max_iter=50000):
    """
    Local search: swap edges to reduce the number of distinct sums/products.
    """
    if time.time() > _DEADLINE:
        return edges, _score_fast(values, edges)
    
    cur_edges = set(edges)
    cur_score = _score_fast(values, cur_edges)
    cur_sums, cur_prods = _score_components(values, cur_edges)
    
    best_edges = set(cur_edges)
    best_score = cur_score
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(all_possible_edges)
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9998
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 2:
            break
        
        # Try an edge swap
        if len(cur_edges) == 0:
            continue
        
        edge_to_remove = random.choice(list(cur_edges))
        available = [e for e in all_possible_edges if e not in cur_edges]
        if not available:
            continue
        edge_to_add = random.choice(available)
        
        new_edges = cur_edges - {edge_to_remove} | {edge_to_add}
        new_score = _score_fast(values, new_edges)
        
        delta = new_score - cur_score
        if delta <= 0 or random.random() < math.exp(-delta / T):
            cur_edges = new_edges
            cur_score = new_score
            cur_sums, cur_prods = _score_components(values, cur_edges)
            if cur_score < best_score:
                best_score = cur_score
                best_edges = set(cur_edges)
        
        T = max(T_min, T * cooling_rate)
    
    return list(best_edges), best_score

def _value_mutation(values, n):
    """
    Mutate values to create more collisions in sums and products.
    """
    new_values = list(values)
    idx = random.randint(0, n - 1)
    strategy = random.random()
    
    if strategy < 0.25:
        # Small perturbation
        delta = random.randint(-5, 5)
        new_values[idx] = max(1, min(10000, new_values[idx] + delta))
    elif strategy < 0.5:
        # Replace with random small value
        new_values[idx] = random.randint(1, 30)
    elif strategy < 0.75:
        # Try to match a sum with another vertex
        other_idx = random.choice([i for i in range(n) if i != idx])
        target_sum = random.choice([values[other_idx] + values[random.randint(0, n-1)] for _ in range(5)])
        new_values[idx] = max(1, min(10000, target_sum - values[other_idx]))
    else:
        # Replace with a divisor-rich value
        candidates = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
        new_values[idx] = random.choice(candidates)
    
    return new_values

def _joint_local_search(values, edges, n, m, max_iter=80000):
    """
    Joint local search over both values and edges.
    """
    if time.time() > _DEADLINE:
        return values, edges, _score_fast(values, edges)
    
    cur_values = list(values)
    cur_edges = set(edges)
    cur_score = _score_fast(cur_values, cur_edges)
    
    best_values = list(cur_values)
    best_edges = set(cur_edges)
    best_score = cur_score
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    T = 8.0
    T_min = 0.05
    cooling_rate = 0.9997
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 2:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Edge swap
            if len(cur_edges) == 0:
                continue
            edge_to_remove = random.choice(list(cur_edges))
            available = [e for e in all_possible_edges if e not in cur_edges]
            if not available:
                continue
            edge_to_add = random.choice(available)
            
            new_edges = cur_edges - {edge_to_remove} | {edge_to_add}
            new_score = _score_fast(cur_values, new_edges)
            
            delta = new_score - cur_score
            if delta <= 0 or random.random() < math.exp(-delta / T):
                cur_edges = new_edges
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = set(cur_edges)
                    best_values = list(cur_values)
                    
        elif move_type < 0.8:
            # Value mutation
            new_values = _value_mutation(cur_values, n)
            if len(set(new_values)) != n:
                continue
            if min(new_values) < 1 or max(new_values) > 10000:
                continue
            
            new_score = _score_fast(new_values, cur_edges)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < math.exp(-delta / T):
                cur_values = new_values
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_values = list(cur_values)
                    best_edges = set(cur_edges)
                    
        else:
            # Simultaneous: mutate one value and swap one edge
            new_values = _value_mutation(cur_values, n)
            if len(set(new_values)) != n:
                continue
            if min(new_values) < 1 or max(new_values) > 10000:
                continue
            
            if len(cur_edges) == 0:
                continue
            edge_to_remove = random.choice(list(cur_edges))
            available = [e for e in all_possible_edges if e not in cur_edges]
            if not available:
                continue
            edge_to_add = random.choice(available)
            new_edges = cur_edges - {edge_to_remove} | {edge_to_add}
            
            new_score = _score_fast(new_values, new_edges)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < math.exp(-delta / T):
                cur_values = new_values
                cur_edges = new_edges
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_values = list(cur_values)
                    best_edges = set(cur_edges)
        
        T = max(T_min, T * cooling_rate)
    
    return best_values, list(best_edges), best_score

def _generate_value_candidates(n):
    """Generate diverse value sets."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Divisor-rich values
    divisor_rich = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    candidates.append(divisor_rich[:n])
    
    # 4. Powers of 2
    candidates.append([2**i for i in range(n)])
    
    # 5. Mixed small values with structure
    random.seed(42)
    for _ in range(30):
        vals = sorted(random.sample(range(1, 40), n))
        candidates.append(vals)
    
    # 6. Values with many common divisors
    for base in [2, 3, 4, 5, 6]:
        vals = []
        for i in range(n):
            v = base * (i + 1)
            if v <= 10000:
                vals.append(v)
        if len(vals) == n:
            candidates.append(vals)
    
    # 7. Arithmetic progressions
    for start in range(1, 10):
        for step in range(1, 8):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= 10000:
                candidates.append(vals)
    
    # 8. Geometric progressions
    for base in [2, 3]:
        vals = [base ** i for i in range(n)]
        if max(vals) <= 10000:
            candidates.append(vals)
    
    # 9. Random with bias toward divisor-rich numbers
    divisor_pool = [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 27, 28, 30,
                    32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 81,
                    84, 90, 96, 98, 99, 100]
    for _ in range(20):
        vals = sorted(random.sample(divisor_pool, min(n, len(divisor_pool))))
        if len(vals) == n:
            candidates.append(vals)
    
    # 10. Pairs that create product collisions
    # e.g., (2,6),(3,4) -> 12; (2,12),(3,8),(4,6) -> 24; etc.
    structured = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
    candidates.append(structured[:n])
    
    # 11. Values around a common sum
    for target in range(20, 50):
        vals = []
        used = set()
        for i in range(n):
            a = random.randint(1, target - 2)
            b = target - a
            if a not in used and b not in used and a != b and 1 <= a <= 10000 and 1 <= b <= 10000:
                vals.extend([a, b])
                used.add(a)
                used.add(b)
                if len(vals) >= n:
                    break
        if len(set(vals)) >= n:
            vals = sorted(list(set(vals))[:n])
            if len(vals) == n and min(vals) >= 1 and max(vals) <= 10000:
                candidates.append(vals)
    
    return candidates

def _polish(values, edges, n, m, max_iter=20000):
    """
    High-precision polish: try all single-edge swaps and single-value changes.
    """
    if time.time() > _DEADLINE:
        return values, edges, _score_fast(values, edges)
    
    cur_values = list(values)
    cur_edges = set(edges)
    cur_score = _score_fast(cur_values, cur_edges)
    best_values = list(cur_values)
    best_edges = set(cur_edges)
    best_score = cur_score
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 2:
            break
        
        improved = False
        
        # Try all edge swaps (remove one, add one)
        for edge_to_remove in list(cur_edges):
            if time.time() > _DEADLINE:
                break
            for edge_to_add in all_possible_edges:
                if edge_to_add in cur_edges:
                    continue
                new_edges = cur_edges - {edge_to_remove} | {edge_to_add}
                new_score = _score_fast(cur_values, new_edges)
                if new_score < cur_score:
                    cur_edges = new_edges
                    cur_score = new_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(cur_values)
                        best_edges = set(cur_edges)
                    break
            if improved:
                break
        
        if not improved:
            # Try single value changes
            for idx in range(n):
                if time.time() > _DEADLINE:
                    break
                old_val = cur_values[idx]
                for new_val in range(1, 50):
                    if new_val == old_val:
                        continue
                    new_values = list(cur_values)
                    new_values[idx] = new_val
                    if len(set(new_values)) != n:
                        continue
                    new_score = _score_fast(new_values, cur_edges)
                    if new_score < cur_score:
                        cur_values = new_values
                        cur_score = new_score
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_values = list(cur_values)
                            best_edges = set(cur_edges)
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_values, list(best_edges), best_score

def construct():
    n = 14
    m = 35
    
    best_score = m + 1
    best_result = None
    
    candidates = _generate_value_candidates(n)
    
    random.seed(12345)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > 10000:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        remaining = _DEADLINE - time.time()
        if remaining < 3:
            break
        
        # Phase 1: Greedy edge selection
        edges = _greedy_edge_selection(vals, m, n)
        if len(edges) != m:
            continue
        
        score = _score_fast(vals, edges)
        
        # Phase 2: Joint local search
        remaining = _DEADLINE - time.time()
        if remaining < 2:
            break
        max_iter = min(60000, int(remaining * 1500))
        max_iter = max(max_iter, 5000)
        
        vals2, edges2, score2 = _joint_local_search(vals, edges, n, m, max_iter)
        
        # Phase 3: Polish
        remaining = _DEADLINE - time.time()
        if remaining < 1:
            break
        max_polish = min(15000, int(remaining * 800))
        max_polish = max(max_polish, 2000)
        
        vals3, edges3, score3 = _polish(vals2, edges2, n, m, max_polish)
        
        if score3 < best_score:
            best_score = score3
            best_result = (vals3, edges3)
            if best_score <= 2:
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
