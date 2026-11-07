import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 14
    m = 35
    
    # Candidate value sets
    # 1. Standard 1..14
    # 2. Slightly modified to potentially increase collisions
    # 3. Randomized small integers
    
    candidate_sets = []
    base = list(range(1, n + 1))
    candidate_sets.append(base)
    
    # Try a few random sets of small distinct integers
    # The values should be small to keep sums/products small and overlapping.
    # Max value 10000 is allowed, but small is better for collisions.
    # Let's try sets drawn from 1..30
    random.seed(42)
    for _ in range(50):
        vals = random.sample(range(1, 31), n)
        candidate_sets.append(vals)
        
    # Also try sets with more structure, e.g., pairs that sum to a constant
    # e.g., 1, 14, 2, 13, ... is already in 1..14
    
    best_score = float('inf')
    best_result = None
    
    for vals in candidate_sets:
        if time.time() > deadline:
            break
            
        # Precompute sums and products for all pairs
        all_edges = list(itertools.combinations(range(n), 2))
        edge_sums = []
        edge_prods = []
        for i, j in all_edges:
            edge_sums.append(vals[i] + vals[j])
            edge_prods.append(vals[i] * vals[j])
            
        # Greedy edge selection
        selected_indices = []
        seen_sums = set()
        seen_prods = set()
        
        remaining = list(range(len(all_edges)))
        
        for _ in range(m):
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_new_sums = float('inf')
            best_new_prods = float('inf')
            best_edge_idx = -1
            
            for idx in remaining:
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                new_sums = 0 if s in seen_sums else 1
                new_prods = 0 if p in seen_prods else 1
                
                # We want to minimize the max(distinct_sums, distinct_prods)
                # Current distinct sums = len(seen_sums)
                # Current distinct prods = len(seen_prods)
                # New distinct sums = len(seen_sums) + new_sums
                # New distinct prods = len(seen_prods) + new_prods
                # We want to minimize max(len(seen_sums) + new_sums, len(seen_prods) + new_prods)
                
                curr_max = max(len(seen_sums), len(seen_prods))
                new_max = max(len(seen_sums) + new_sums, len(seen_prods) + new_prods)
                
                # Tie breaker: prefer adding to the set that is currently smaller?
                # Or just minimize the increase in the max.
                # Let's use a tuple for comparison: (new_max, new_sums, new_prods)
                
                if best_idx == -1:
                    best_idx = idx
                    best_new_sums = new_sums
                    best_new_prods = new_prods
                    best_edge_idx = idx
                    best_new_max = new_max
                else:
                    if new_max < best_new_max:
                        best_idx = idx
                        best_new_sums = new_sums
                        best_new_prods = new_prods
                        best_edge_idx = idx
                        best_new_max = new_max
                    elif new_max == best_new_max:
                        # Tie break: minimize total new additions
                        if (new_sums + new_prods) < (best_new_sums + best_new_prods):
                            best_idx = idx
                            best_new_sums = new_sums
                            best_new_prods = new_prods
                            best_edge_idx = idx
                            best_new_max = new_max
                            # Further tie break: prefer sums over prods? Or vice versa?
                            # Let's just take the first one found with minimal sum+prod
                
            if best_idx == -1:
                break
                
            # Add the best edge
            selected_indices.append(all_edges[best_idx])
            seen_sums.add(edge_sums[best_idx])
            seen_prods.add(edge_prods[best_idx])
            remaining.remove(best_idx)
            
        if len(selected_indices) < m:
            continue
            
        current_score = max(len(seen_sums), len(seen_prods))
        
        if current_score < best_score:
            best_score = current_score
            best_result = {
                'values': vals,
                'edges': [list(e) for e in selected_indices]
            }
            
    if best_result is None:
        # Fallback to baseline
        best_result = {
            'values': list(range(1, n + 1)),
            'edges': [list(e) for e in list(itertools.combinations(range(n), 2))[:m]]
        }
        
    return best_result
