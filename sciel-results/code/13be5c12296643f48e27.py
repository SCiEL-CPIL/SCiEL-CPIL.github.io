# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Use smallest values to minimize range and maximize overlap
    values = list(range(1, n + 1))
    
    # All possible edges
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Precompute sum and product for all possible edges
    edge_sums = [0] * num_all_edges
    edge_prods = [0] * num_all_edges
    for i, (u, v) in enumerate(all_edges):
        edge_sums[i] = values[u] + values[v]
        edge_prods[i] = values[u] * values[v]
        
    # Initialize counts
    max_sum_val = max(edge_sums)
    max_prod_val = max(edge_prods)
    
    def greedy_init(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        # Start with empty set
        current_indices = []
        current_set = set()
        sum_counts = [0] * (max_sum_val + 1)
        prod_counts = [0] * (max_prod_val + 1)
        distinct_sums = 0
        distinct_prods = 0
        
        for _ in range(m):
            best_idx = -1
            best_score = float('inf')
            best_ds = 0
            best_dp = 0
            
            # To add diversity, we can randomize the order of candidates or tie-breaking
            candidates = list(range(num_all_edges))
            random.shuffle(candidates)
            
            for idx in candidates:
                if idx in current_set:
                    continue
                
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                # Simulate adding this edge
                new_ds = distinct_sums + (1 if sum_counts[s] == 0 else 0)
                new_dp = distinct_prods + (1 if prod_counts[p] == 0 else 0)
                
                score = max(new_ds, new_dp)
                
                # Tie-breaker: prefer edges that don't increase distinct counts
                # Or prefer edges with higher counts (more "popular")
                tie_breaker = -(sum_counts[s] + prod_counts[p])
                
                if (score, tie_breaker) < (best_score, best_score): # Rough comparison
                     # We want to minimize score. If score is same, we want higher tie_breaker (less negative)
                     # Actually, let's just use a tuple: (score, -tie_breaker)
                     pass
                
                # Let's use a cleaner comparison
                current_candidate_key = (score, -tie_breaker)
                if best_idx == -1:
                    best_idx = idx
                    best_score = score
                    best_key = current_candidate_key
                    best_ds = new_ds
                    best_dp = new_dp
                else:
                    if current_candidate_key < best_key:
                        best_idx = idx
                        best_score = score
                        best_key = current_candidate_key
                        best_ds = new_ds
                        best_dp = new_dp
            
            # Apply best edge
            if best_idx != -1:
                current_indices.append(best_idx)
                current_set.add(best_idx)
                s = edge_sums[best_idx]
                p = edge_prods[best_idx]
                if sum_counts[s] == 0:
                    distinct_sums += 1
                sum_counts[s] += 1
                if prod_counts[p] == 0:
                    distinct_prods += 1
                prod_counts[p] += 1
            else:
                break # Should not happen
            
        return current_indices, current_set, sum_counts, prod_counts, distinct_sums, distinct_prods

    best_overall_max = float('inf')
    best_overall_edges = []
    
    # Try a few different greedy initializations
    num_starts = 10
    for k in range(num_starts):
        if time.time() > deadline:
            break
            
        current_indices, current_set, sum_counts, prod_counts, distinct_sums, distinct_prods = greedy_init(random_seed=k)
        current_max = max(distinct_sums, distinct_prods)
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall_edges = [list(all_edges[i]) for i in current_indices]
            
        # Local Search
        no_improvement_count = 0
        max_no_improvement = 50 # Lower threshold to allow more restarts if stuck
        
        while time.time() < deadline:
            best_swap = None
            best_new_max = current_max
            best_swap_key = None
            
            # Iterate over all edges to remove
            for i_out in range(m):
                idx_out = current_indices[i_out]
                
                # Remove idx_out
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]
                
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    distinct_sums -= 1
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    distinct_prods -= 1
                    
                # Try adding all non-edges
                for idx_in in range(num_all_edges):
                    if idx_in in current_set:
                        continue
                        
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    if sum_counts[s_in] == 0:
                        new_distinct_sums = distinct_sums + 1
                    else:
                        new_distinct_sums = distinct_sums
                        
                    if prod_counts[p_in] == 0:
                        new_distinct_prods = distinct_prods + 1
                    else:
                        new_distinct_prods = distinct_prods
                    
                    new_max = max(new_distinct_sums, new_distinct_prods)
                    
                    # Tie-breaker for swaps: prefer swaps that maintain counts or reduce them
                    # Key: (new_max, new_distinct_sums + new_distinct_prods)
                    swap_key = (new_max, new_distinct_sums + new_distinct_prods)
                    
                    if best_swap is None:
                        best_swap = (i_out, idx_in)
                        best_new_max = new_max
                        best_swap_key = swap_key
                    else:
                        if swap_key < best_swap_key:
                            best_swap = (i_out, idx_in)
                            best_new_max = new_max
                            best_swap_key = swap_key
                            
                # Revert removal of idx_out
                sum_counts[s_out] += 1
                if sum_counts[s_out] == 1:
                    distinct_sums += 1
                prod_counts[p_out] += 1
                if prod_counts[p_out] == 1:
                    distinct_prods += 1
                    
            if best_swap:
                i_out, idx_in = best_swap
                idx_out = current_indices[i_out]
                
                # Apply swap
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    distinct_sums -= 1
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    distinct_prods -= 1
                
                s_in = edge_sums[idx_in]
                p_in = edge_prods[idx_in]
                if sum_counts[s_in] == 0:
                    distinct_sums += 1
                sum_counts[s_in] += 1
                if prod_counts[p_in] == 0:
                    distinct_prods += 1
                prod_counts[p_in] += 1
                
                current_indices[i_out] = idx_in
                current_set.remove(idx_out)
                current_set.add(idx_in)
                
                current_max = max(distinct_sums, distinct_prods)
                
                if current_max < best_overall_max:
                    best_overall_max = current_max
                    best_overall_edges = [list(all_edges[i]) for i in current_indices]
                    no_improvement_count = 0
                else:
                    # Check if this swap was actually an improvement over the *current* state before swap?
                    # The local search finds the best swap relative to the current state.
                    # If best_new_max < current_max, it's an improvement.
                    if best_new_max < current_max:
                        no_improvement_count = 0
                    else:
                        # Sideways move?
                        # If we allow sideways moves, we might escape local minima.
                        # But strictly, if no improvement, we count no_improvement.
                        no_improvement_count += 1
            else:
                no_improvement_count += 1
                
            if no_improvement_count > max_no_improvement:
                # Break to next restart
                break
                
    # Return the best found
    # Ensure types are correct: list of ints, list of lists of ints
    return {
        'values': values,
        'edges': best_overall_edges
    }
# EVOLVE-BLOCK-END
