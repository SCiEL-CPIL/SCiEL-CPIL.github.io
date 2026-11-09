"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict

_DEADLINE = time.time() + 85  # safety margin before 90s kill

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

def _precompute_all_edges(n):
    """Precompute all possible edges for n vertices."""
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    return all_edges

def _edge_index_map(n):
    """Create a mapping from (i,j) tuple to index in the all_edges list."""
    all_edges = _precompute_all_edges(n)
    return {e: idx for idx, e in enumerate(all_edges)}

def _generate_structured_values(n, max_val=10000):
    """Generate diverse structured value sets that promote sum/product collisions."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Arithmetic progressions
    for start in range(1, 10):
        for step in range(1, 6):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and min(vals) >= 1:
                candidates.append(vals)
    
    # 4. Values with additive structure (many pairs sum to same value)
    # e.g., for sum S, pairs (a, S-a)
    for S in range(2 * n, 2 * n + 20):
        vals = []
        for i in range(n):
            if i < n // 2:
                vals.append(i + 1)
            else:
                vals.append(S - (i - n // 2 + 1))
        if len(set(vals)) == n and min(vals) >= 1 and max(vals) <= max_val:
            candidates.append(sorted(vals))
    
    # 5. Multiplicative structure
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 6. Random with bias toward small values
    random.seed(42)
    for _ in range(20):
        vals = sorted(random.sample(range(1, 50), n))
        candidates.append(vals)
    
    # 7. Fibonacci-like
    candidates.append([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
    
    # 8. Narrow range
    for lo in range(1, 15):
        hi = lo + 8
        if hi - lo + 1 >= n:
            vals = random.sample(range(lo, hi + 1), n)
            if len(vals) == n:
                candidates.append(sorted(vals))
    
    # 9. Pairs that multiply to same value
    # e.g., (2,6),(3,4) -> 12; (2,12),(3,8),(4,6) -> 24
    base_vals = [1, 2, 3, 4, 6, 8, 12, 24, 48, 96, 192, 384, 768, 1536]
    candidates.append(base_vals)
    
    # 10. Mixed small and medium
    for _ in range(10):
        small = random.sample(range(1, 20), n // 2)
        medium = random.sample(range(20, 100), n - n // 2)
        vals = sorted(small + medium)
        if len(set(vals)) == n:
            candidates.append(vals)
    
    return candidates

def _simulated_annealing(values, edges, n, m, all_edges, edge_idx_map, max_iter=50000):
    """Simulated annealing over both values and edges."""
    if time.time() > _DEADLINE:
        return None
    
    cur_values = list(values)
    cur_edges = set(edges)
    cur_score = _score_fast(cur_values, cur_edges)
    
    best_values = list(cur_values)
    best_edges = set(cur_edges)
    best_score = cur_score
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 2:
            break
        
        # Choose move type: 60% edge swap, 40% value mutation
        if random.random() < 0.6:
            # Edge swap: remove one edge, add one
            if len(cur_edges) == 0 or len(all_edges) == len(cur_edges):
                continue
            
            edge_to_remove = random.choice(list(cur_edges))
            available = [e for e in all_edges if e not in cur_edges]
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
        else:
            # Value mutation: change one or two values
            num_mutate = random.choice([1, 2])
            new_values = list(cur_values)
            
            for _ in range(num_mutate):
                idx = random.randint(0, n - 1)
                # Try different mutation strategies
                strategy = random.random()
                if strategy < 0.3:
                    # Small perturbation
                    delta = random.randint(-3, 3)
                    new_val = new_values[idx] + delta
                elif strategy < 0.6:
                    # Replace with random small value
                    new_val = random.randint(1, 50)
                elif strategy < 0.8:
                    # Replace with value that creates collision
                    # Pick a random other vertex and try to match sum or product
                    other_idx = random.choice([i for i in range(n) if i != idx])
                    other_val = new_values[other_idx]
                    # Try to make sum equal to a common sum
                    target_sum = random.randint(2, 100)
                    new_val = target_sum - other_val
                else:
                    # Replace with random value
                    new_val = random.randint(1, 200)
                
                new_val = max(1, min(10000, new_val))
                new_values[idx] = new_val
            
            # Check validity
            if len(set(new_values)) != n:
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
        
        T = max(T_min, T * cooling_rate)
    
    return best_score, best_values, list(best_edges)

def _try_candidate(values, n, m, all_edges, edge_idx_map, max_iter=30000):
    """Try a candidate value set with simulated annealing."""
    if time.time() > _DEADLINE:
        return None
    
    # Initialize edges: start with most "popular" edges
    edge_data = []
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_data.append((i, j, s, p))
    
    sum_counts = defaultdict(int)
    prod_counts = defaultdict(int)
    for i, j, s, p in edge_data:
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    edge_popularity = []
    for idx, (i, j, s, p) in enumerate(edge_data):
        edge_popularity.append(sum_counts[s] + prod_counts[p])
    
    sorted_indices = sorted(range(len(edge_data)), key=lambda x: -edge_popularity[x])
    init_edges = [edge_data[idx][:2] for idx in sorted_indices[:m]]
    
    result = _simulated_annealing(values, init_edges, n, m, all_edges, edge_idx_map, max_iter)
    return result

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = _precompute_all_edges(n)
    edge_idx_map = _edge_index_map(n)
    
    best_score = m + 1
    best_result = None
    
    candidates = _generate_structured_values(n, max_val)
    
    random.seed(12345)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        remaining = _DEADLINE - time.time()
        if remaining < 2:
            break
        max_iter = min(20000, int(remaining * 2000))
        max_iter = max(max_iter, 1000)
        
        result = _try_candidate(vals, n, m, all_edges, edge_idx_map, max_iter)
        if result is None:
            continue
        score, values, edges = result
        if score < best_score:
            best_score = score
            best_result = (values, edges)
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
