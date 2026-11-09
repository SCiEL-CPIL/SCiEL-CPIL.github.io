import time
import random
import math
from itertools import combinations

# Deadline guard: 90s limit, use 85s for safety
START_TIME = time.time()
DEADLINE = START_TIME + 85.0

def _check_time():
    return time.time() > DEADLINE

def _score(values, edges):
    """Calculate the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _optimize_edges(values, n_edges=35, max_iters=5000):
    """
    Find the best set of n_edges edges for a given set of values
    using Simulated Annealing.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    # Precompute sums and products for all possible edges
    edge_props = {}
    for e in all_edges:
        u, v = e
        edge_props[e] = (values[u] + values[v], values[u] * values[v])
        
    # Initialize with a greedy approach
    # Sort edges by how 'common' their sum and product are
    sum_counts = {}
    prod_counts = {}
    for s, p in edge_props.values():
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    def edge_priority(e):
        s, p = edge_props[e]
        return (sum_counts[s], prod_counts[p])
        
    sorted_edges = sorted(all_edges, key=edge_priority, reverse=True)
    
    current_edges = set(sorted_edges[:n_edges])
    current_score = _score(values, list(current_edges))
    
    best_edges = set(current_edges)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    
    iter_count = 0
    while T > min_T and iter_count < max_iters:
        if _check_time():
            break
            
        # Pick a random edge to remove
        u, v = random.choice(list(current_edges))
        
        # Pick a random edge to add (not currently in set)
        # Efficiently pick from non-edges
        # For small n, just sampling is fine
        cand_e = None
        attempts = 0
        while cand_e is None or cand_e in current_edges:
            cand_e = random.choice(all_edges)
            attempts += 1
            if attempts > 100:
                break
                
        if cand_e is None or cand_e in current_edges:
            continue
            
        # Calculate new score
        # We can optimize this by tracking sets, but recalculating is fast for n=14
        # Let's try a faster incremental update if possible, or just recalc if it's cheap enough
        # Recalculation: 35 edges * 2 ops = 70 ops. Very cheap.
        
        new_edges = list(current_edges)
        new_edges.remove((u, v))
        new_edges.append(cand_e)
        
        new_score = _score(values, new_edges)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_edges.discard((u, v))
            current_edges.add(cand_e)
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                
        T *= cooling
        iter_count += 1
        
    return best_score, list(best_edges)

def _generate_value_candidates():
    """Generate diverse candidate value sets."""
    candidates = []
    n = 14
    
    # 1. Small integers with multiplicative structure
    # Powers of 2, 3, 5 mixed
    base_sets = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2 (products very distinct, sums distinct) -> Bad for products? No, 2^a * 2^b = 2^(a+b). Sums 2^a+2^b are distinct.
        # Actually, powers of 2 have very few collisions in sums and products.
        # We want MANY collisions.
        # So we want numbers that share factors.
    ]
    
    # Better candidates: Numbers composed of small primes
    # e.g. 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20
    # Let's generate sets based on "density" of small factors
    
    # Set 1: 1 to 14
    candidates.append(list(range(1, 15)))
    
    # Set 2: 1 to 15, skip 11, 13 (primes cause unique products)
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16])
    
    # Set 3: Multiples of 2 and 3
    multiples = [i for i in range(1, 50) if i % 2 == 0 or i % 3 == 0]
    if len(multiples) >= 14:
        candidates.append(multiples[:14])
        
    # Set 4: Random small numbers with high factor overlap
    random.seed(42)
    for _ in range(20):
        # Sample from 1..30
        vals = random.sample(range(1, 31), 14)
        # Sort to be consistent
        vals.sort()
        candidates.append(vals)
        
    # Set 5: Specific structure for sums (AP)
    candidates.append(list(range(1, 15, 1))) # 1..14
    candidates.append(list(range(2, 30, 2))) # 2, 4, ..., 28
    
    # Set 6: Specific structure for products (Powers of 2 and 3)
    # 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(list(c))
            
    return unique_candidates

def construct():
    n = 14
    m = 35
    
    candidates = _generate_value_candidates()
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Iterate through candidates
    # We can also do a local search on the values themselves
    # But first, let's see what we get with fixed value sets and optimized edges
    
    for vals in candidates:
        if _check_time():
            break
            
        # Ensure valid values
        if len(set(vals)) != n:
            continue
        if min(vals) < 1 or max(vals) > 10000:
            continue
            
        score, edges = _optimize_edges(vals, n_edges=m, max_iters=2000)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
            
            # If we hit a very low score, we can stop early
            if best_score <= 5:
                break
                
    # If we have time, try to improve the best values found
    if best_values is not None and not _check_time():
        current_vals = best_values[:]
        current_edges = best_edges
        current_score = best_score
        
        # Local search on values
        # Swap a value with a random small integer
        for _ in range(1000):
            if _check_time():
                break
                
            idx = random.randint(0, n - 1)
            old_val = current_vals[idx]
            new_val = random.randint(1, 50)
            
            if new_val == old_val:
                continue
            if new_val in current_vals:
                continue
                
            current_vals[idx] = new_val
            
            # Re-optimize edges for this new value set
            # Use fewer iterations for speed
            new_score, new_edges = _optimize_edges(current_vals, n_edges=m, max_iters=500)
            
            if new_score < current_score:
                current_score = new_score
                best_score = new_score
                best_values = current_vals[:]
                best_edges = new_edges
            else:
                # Revert
                current_vals[idx] = old_val
                
            if best_score <= 5:
                break

    # Final validation
    if best_values is None:
        # Fallback
        best_values = list(range(1, n + 1))
        best_edges = list(combinations(range(n), 2))[:m]
        
    # Ensure types
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Sanity checks
    assert len(values) == n
    assert len(set(values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert min(values) >= 1 and max(values) <= 10000
    assert len(edges) == m
    assert all(u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': values, 'edges': edges}
