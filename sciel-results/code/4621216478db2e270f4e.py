import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 14
    m = 35
    maximum = 10000
    
    # Candidate value sets
    # 1. Small integers 1-14
    vals_1 = list(range(1, 15))
    # 2. Multiplicative rich set (small numbers with many factors)
    vals_2 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72]
    # 3. Another AP with different step
    vals_3 = [5 * (i + 1) for i in range(14)]
    # 4. Random small integers
    random.seed(42)
    vals_4 = random.sample(range(1, 100), 14)
    
    candidates = [vals_1, vals_2, vals_3, vals_4]
    
    # Filter valid candidates
    valid_candidates = []
    for v in candidates:
        if len(set(v)) != n:
            continue
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            continue
        if not all(1 <= x <= maximum for x in v):
            continue
        valid_candidates.append(v)
    
    if not valid_candidates:
        # Fallback
        valid_candidates = [list(range(1, 15))]

    best_score = float('inf')
    best_result = None
    
    all_possible_edges = [list(e) for e in combinations(range(n), 2)]
    num_edges_total = len(all_possible_edges) # 91
    
    for vals in valid_candidates:
        # Precompute sums and products for all edges
        edge_sums = [0] * num_edges_total
        edge_prods = [0] * num_edges_total
        for idx, (i, j) in enumerate(all_possible_edges):
            edge_sums[idx] = vals[i] + vals[j]
            edge_prods[idx] = vals[i] * vals[j]
            
        # Count frequency of each product to guide initial selection
        prod_freq = defaultdict(int)
        for p in edge_prods:
            prod_freq[p] += 1
            
        # Sort edge indices by product frequency (descending)
        # Ties broken by sum frequency? Let's just do product freq first.
        # To minimize distinct products, we want to pick edges that share products.
        sorted_edge_indices = sorted(range(num_edges_total), key=lambda x: prod_freq[edge_prods[x]], reverse=True)
        
        # Initial selection: top 35 edges by product frequency
        initial_indices = set(sorted_edge_indices[:m])
        
        # Local Search
        # We will try to improve the solution by swapping edges.
        
        def get_score(indices_set):
            sums = set()
            prods = set()
            for idx in indices_set:
                sums.add(edge_sums[idx])
                prods.add(edge_prods[idx])
            return max(len(sums), len(prods))
        
        # Run local search for this value set
        # We do multiple restarts
        num_restarts = 10
        max_iters = 2000
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            if r == 0:
                current_indices = set(initial_indices)
            else:
                # Random restart
                current_indices = set(random.sample(range(num_edges_total), m))
            
            current_score = get_score(current_indices)
            
            # Track frequencies for O(1) updates
            sum_counts = defaultdict(int)
            prod_counts = defaultdict(int)
            for idx in current_indices:
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1
                
            distinct_sums = len(sum_counts)
            distinct_prods = len(prod_counts)
            
            improved = True
            iter_count = 0
            while improved and iter_count < max_iters and time.time() < deadline - 5.0:
                improved = False
                iter_count += 1
                
                # Pick a random edge to remove
                idx_out = random.choice(list(current_indices))
                
                # Pick a random edge to add (not currently in set)
                # To speed up, we can precompute a list of available edges, but it changes.
                # Simple way: try a few random candidates for idx_in
                found_swap = False
                for _ in range(20): # Try up to 20 random candidates for the incoming edge
                    idx_in = random.randint(0, num_edges_total - 1)
                    if idx_in in current_indices:
                        continue
                    
                    # Simulate swap
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    # Update sum counts
                    sum_counts[s_out] -= 1
                    if sum_counts[s_out] == 0:
                        del sum_counts[s_out]
                        new_distinct_sums = distinct_sums - 1
                    else:
                        new_distinct_sums = distinct_sums
                        
                    sum_counts[s_in] += 1
                    if s_in not in sum_counts or sum_counts[s_in] == 1: # Just added
                         # If it was 0 before, distinct count increases
                         # Wait, if s_in was not in dict, it's new.
                         # If it was in dict, count was >=1, so distinct count doesn't change.
                         # Logic:
                         # Before: count(s_in) = c. After: c+1.
                         # If c == 0 (not in dict), distinct increases by 1.
                         # If c > 0, distinct stays same.
                         pass 
                    # Let's re-evaluate distinct counts properly
                    # It's safer to just calculate the change in distinct counts directly.
                    
                    # Restore counts for simulation rollback if needed, or just compute delta
                    # Let's compute delta_distinct_sums
                    delta_ds = 0
                    if sum_counts[s_out] == 0: # s_out is gone
                        delta_ds -= 1
                    # Note: we already decremented sum_counts[s_out]
                    # Now check s_in
                    # Before adding s_in, what was its count?
                    # We incremented it. 
                    # If it was 0 before, it's 1 now. Distinct +1.
                    # If it was >0 before, it's >1 now. Distinct +0.
                    # We need to know the count BEFORE the increment.
                    # Current count is sum_counts[s_in].
                    # If sum_counts[s_in] == 1, it was 0 before.
                    if sum_counts[s_in] == 1:
                        delta_ds += 1
                        
                    new_distinct_sums = distinct_sums + delta_ds
                    
                    # Do the same for products
                    prod_counts[p_out] -= 1
                    delta_dp = 0
                    if prod_counts[p_out] == 0:
                        delta_dp -= 1
                        
                    prod_counts[p_in] += 1
                    if prod_counts[p_in] == 1:
                        delta_dp += 1
                        
                    new_distinct_prods = distinct_prods + delta_dp
                    
                    new_score = max(new_distinct_sums, new_distinct_prods)
                    
                    if new_score < current_score:
                        # Accept swap
                        current_indices.remove(idx_out)
                        current_indices.add(idx_in)
                        current_score = new_score
                        distinct_sums = new_distinct_sums
                        distinct_prods = new_distinct_prods
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Rollback counts
                        sum_counts[s_out] += 1
                        sum_counts[s_in] -= 1
                        if sum_counts[s_in] == 0:
                            del sum_counts[s_in]
                            
                        prod_counts[p_out] += 1
                        prod_counts[p_in] -= 1
                        if prod_counts[p_in] == 0:
                            del prod_counts[p_in]
                        
                # If no swap found in 20 tries, we might be stuck or just unlucky.
                # We could try a different strategy for idx_in selection, e.g., prefer edges with common products.
                # For now, if no swap, we break the inner loop for this restart?
                # No, we can just continue to next iteration, but if 'improved' is False, the while loop ends.
                # So if no swap found, we exit local search for this restart.
                
            if current_score < best_score:
                best_score = current_score
                best_edges = [all_possible_edges[i] for i in current_indices]
                best_vals = vals

    # Fallback if no valid result found (should not happen)
    if best_result is None:
        best_vals = valid_candidates[0]
        best_edges = all_possible_edges[:m]

    # Final validation
    assert len(best_vals) == n
    assert len(set(best_vals)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in best_vals)
    assert all(1 <= v <= maximum for v in best_vals)
    assert len(best_edges) == m
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {'values': list(best_vals), 'edges': [list(e) for e in best_edges]}
