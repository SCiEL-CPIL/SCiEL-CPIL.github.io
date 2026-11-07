"""Strategy: Joint optimization of value sets and edge selection via Simulated Annealing."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 14
    m = 35
    
    # 1. Generate diverse candidate value sets
    # We want values that create many sum/product collisions.
    # Common structures:
    # - Powers of 2: products are unique (2^a * 2^b = 2^(a+b)), sums are unique (binary rep). Bad for products.
    # - Small integers: many sum collisions (e.g., 1+4=2+3), product collisions (1*6=2*3).
    # - Geometric progressions: a, ar, ar^2... products a^2 r^(i+k). Collisions if i+k same.
    
    candidates = []
    
    # Strategy A: Small consecutive integers
    for start in range(1, 10):
        for step in range(1, 5):
            vals = [start + i*step for i in range(n)]
            if all(v <= 10000 and v > 0 for v in vals) and len(set(vals)) == n:
                candidates.append(vals)
                
    # Strategy B: Geometric progressions
    for base in range(2, 10):
        for ratio in [2, 3, 4, 5, 6]:
            vals = []
            v = base
            for _ in range(n):
                if v > 10000: break
                vals.append(v)
                v *= ratio
            if len(vals) == n:
                candidates.append(vals)
                
    # Strategy C: Mixed small primes/numbers to encourage product collisions
    # E.g., numbers with many factors
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # Strategy D: Random search for "collision-rich" sets
    # A set is good if it has many pairs with same sum or product.
    # We'll just use a few random sets within small range.
    for _ in range(20):
        if time.time() > deadline - 5: break
        vals = sorted(random.sample(range(1, 100), n))
        if len(set(vals)) == n:
            candidates.append(vals)

    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    best_raw = float('inf')
    best_result = None

    # Pre-compute all pairs
    all_pairs = list(itertools.combinations(range(n), 2))
    
    # For each candidate value set, run Simulated Annealing to find best 35 edges
    for vals in unique_candidates:
        if time.time() > deadline - 2:
            break
            
        # Pre-compute sums and products for all pairs
        pair_sums = {}
        pair_prods = {}
        for u, v in all_pairs:
            s = vals[u] + vals[v]
            p = vals[u] * vals[v]
            pair_sums[(u, v)] = s
            pair_prods[(u, v)] = p
            
        # Group pairs by sum and product to know "collision potential"
        # If we pick an edge, it contributes to the count of that sum/product.
        # We want to minimize distinct sums and distinct products.
        
        # Simulated Annealing State:
        # - selected_edges: set of 35 tuples (u, v)
        # - distinct_sums: set of sums for selected edges
        # - distinct_prods: set of products for selected edges
        # - raw: max(len(distinct_sums), len(distinct_prods))
        
        # Initialize: Random 35 edges
        current_edges = set(random.sample(all_pairs, m))
        current_sums = set(pair_sums[e] for e in current_edges)
        current_prods = set(pair_prods[e] for e in current_edges)
        current_raw = max(len(current_sums), len(current_prods))
        
        best_edges_in_sa = set(current_edges)
        best_raw_in_sa = current_raw
        
        temp = 50.0
        cooling = 0.995
        steps = 2000
        
        for step in range(steps):
            if time.time() > deadline - 1:
                break
                
            # Move: Remove one random edge, add one random edge
            # Or swap: Remove one, add one.
            e_remove = random.choice(list(current_edges))
            # Pick e_add not in current_edges
            e_add = random.choice(all_pairs)
            while e_add in current_edges:
                e_add = random.choice(all_pairs)
            
            # Calculate delta
            # Removing e_remove:
            s_rm = pair_sums[e_remove]
            p_rm = pair_prods[e_remove]
            
            # Adding e_add:
            s_ad = pair_sums[e_add]
            p_ad = pair_prods[e_add]
            
            # New distinct counts
            # If s_rm was unique in current_sums, len decreases by 1.
            # If s_ad is already in current_sums, len stays. If not, increases by 1.
            
            # Efficient check:
            # Count occurrences? That's expensive.
            # Instead, just recompute sets for the move?
            # 35 edges, set ops are fast.
            
            new_edges = (current_edges - {e_remove}) | {e_add}
            new_sums = set(pair_sums[e] for e in new_edges)
            new_prods = set(pair_prods[e] for e in new_edges)
            new_raw = max(len(new_sums), len(new_prods))
            
            delta = new_raw - current_raw
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_edges = new_edges
                current_sums = new_sums
                current_prods = new_prods
                current_raw = new_raw
                
                if current_raw < best_raw_in_sa:
                    best_raw_in_sa = current_raw
                    best_edges_in_sa = set(current_edges)
            
            temp *= cooling
        
        # End SA for this value set
        if best_raw_in_sa < best_raw:
            best_raw = best_raw_in_sa
            # Construct result
            edges_list = [[int(u), int(v)] for u, v in best_edges_in_sa]
            # Ensure deterministic order? Not required, but good for stability.
            edges_list.sort()
            best_result = {
                'values': [int(x) for x in vals],
                'edges': edges_list
            }

    # If no result found (should not happen), fallback to simple greedy
    if best_result is None:
        vals = list(range(1, 15))
        edges = []
        used = set()
        # Greedy: pick edges that minimize distinct sums
        pair_sums = {}
        for u, v in all_pairs:
            pair_sums[(u,v)] = vals[u] + vals[v]
        # Sort by sum frequency?
        sum_counts = {}
        for e, s in pair_sums.items():
            if s not in sum_counts:
                sum_counts[s] = []
            sum_counts[s].append(e)
        
        for s, es in sorted(sum_counts.items(), key=lambda x: -len(x[1])):
            for e in es:
                if e not in used:
                    edges.append([int(e[0]), int(e[1])])
                    used.add(e)
                    if len(edges) == m:
                        break
            if len(edges) == m:
                break
        
        best_result = {
            'values': [int(x) for x in vals],
            'edges': edges
        }

    return best_result
# EVOLVE-BLOCK-END
