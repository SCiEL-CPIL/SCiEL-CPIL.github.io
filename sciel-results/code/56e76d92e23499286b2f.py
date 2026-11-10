import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    # Candidate value sets - focus on sets with many repeated sums/products
    candidates = [
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 40, 60],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 45, 60, 90],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 6, 9, 12, 18, 27, 36, 54, 72, 90, 108],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24],
        [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40],
        [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75],
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 3, 4, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 18, 20, 24],
        [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22],
    ]
    
    # Filter valid candidates
    valid_candidates = []
    seen = set()
    for v in candidates:
        if len(set(v)) != n:
            continue
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            continue
        if not all(1 <= x <= maximum for x in v):
            continue
        key = tuple(sorted(v))
        if key in seen:
            continue
        seen.add(key)
        valid_candidates.append(v)
    
    if not valid_candidates:
        valid_candidates = [list(range(1, 15))]
    
    best_score = float('inf')
    best_edges = None
    best_vals = None
    
    all_possible_edges = [list(e) for e in combinations(range(n), 2)]
    num_edges_total = len(all_possible_edges)  # 91
    
    random.seed(42)
    
    for vals in valid_candidates:
        if time.time() > deadline - 15.0:
            break
            
        # Precompute sums and products for all edges
        edge_sums = [0] * num_edges_total
        edge_prods = [0] * num_edges_total
        for idx, (i, j) in enumerate(all_possible_edges):
            edge_sums[idx] = vals[i] + vals[j]
            edge_prods[idx] = vals[i] * vals[j]
        
        # Count frequencies
        sum_freq = defaultdict(int)
        prod_freq = defaultdict(int)
        for s in edge_sums:
            sum_freq[s] += 1
        for p in edge_prods:
            prod_freq[p] += 1
        
        # Strategy: Use constraint satisfaction approach
        # For a given K, check if we can find K sum values and K product values
        # such that there exist 35 edges each having a sum in the chosen K sums
        # AND a product in the chosen K products.
        
        # Greedy approach: build edge set incrementally, minimizing increase in max(distinct_sums, distinct_prods)
        
        # Compute a "popularity" score for each edge
        edge_scores = []
        for idx in range(num_edges_total):
            s_score = sum_freq[edge_sums[idx]]
            p_score = prod_freq[edge_prods[idx]]
            combined = min(s_score, p_score)
            edge_scores.append((combined, s_score, p_score, idx))
        
        # Sort by combined score descending
        edge_scores.sort(reverse=True)
        
        # Greedy construction
        selected = []
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        distinct_sums = 0
        distinct_prods = 0
        
        for _, _, _, idx in edge_scores:
            if len(selected) >= m:
                break
            if time.time() > deadline - 10.0:
                break
            
            s = edge_sums[idx]
            p = edge_prods[idx]
            
            new_ds = distinct_sums + (1 if s not in sum_counts else 0)
            new_dp = distinct_prods + (1 if p not in prod_counts else 0)
            
            selected.append(idx)
            if s not in sum_counts:
                distinct_sums += 1
            sum_counts[s] += 1
            if p not in prod_counts:
                distinct_prods += 1
            prod_counts[p] += 1
        
        current_score = max(distinct_sums, distinct_prods)
        
        # Local search to improve
        if time.time() < deadline - 10.0:
            improved = True
            max_iters = 200
            iter_count = 0
            
            while improved and iter_count < max_iters and time.time() < deadline - 10.0:
                improved = False
                iter_count += 1
                
                # Try random swaps
                for _ in range(100):
                    if time.time() > deadline - 10.0:
                        break
                    
                    out_pos = random.randint(0, len(selected) - 1)
                    idx_out = selected[out_pos]
                    
                    selected_set = set(selected)
                    
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    
                    # Find candidate edges that share sum or product with current palette
                    candidates_in = []
                    for idx in range(num_edges_total):
                        if idx in selected_set:
                            continue
                        s_in = edge_sums[idx]
                        p_in = edge_prods[idx]
                        adds_sum = 1 if s_in not in sum_counts else 0
                        adds_prod = 1 if p_in not in prod_counts else 0
                        frees_sum = 1 if sum_counts[s_out] == 1 else 0
                        frees_prod = 1 if prod_counts[p_out] == 1 else 0
                        net_change = (adds_sum - frees_sum) + (adds_prod - frees_prod)
                        if net_change <= 0:
                            candidates_in.append((net_change, sum_freq[s_in], prod_freq[p_in], idx))
                    
                    if not candidates_in:
                        continue
                    
                    candidates_in.sort()
                    _, _, _, idx_in = candidates_in[0]
                    
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    # Update counts
                    sum_counts[s_out] -= 1
                    if sum_counts[s_out] == 0:
                        del sum_counts[s_out]
                    prod_counts[p_out] -= 1
                    if prod_counts[p_out] == 0:
                        del prod_counts[p_out]
                    
                    sum_counts[s_in] += 1
                    prod_counts[p_in] += 1
                    
                    new_ds = len(sum_counts)
                    new_dp = len(prod_counts)
                    new_score = max(new_ds, new_dp)
                    
                    if new_score < current_score:
                        selected[out_pos] = idx_in
                        current_score = new_score
                        improved = True
                        break
                    else:
                        sum_counts[s_in] -= 1
                        if sum_counts[s_in] == 0:
                            del sum_counts[s_in]
                        prod_counts[p_in] -= 1
                        if prod_counts[p_in] == 0:
                            del prod_counts[p_in]
                        
                        sum_counts[s_out] += 1
                        prod_counts[p_out] += 1
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [all_possible_edges[i] for i in selected]
            best_vals = vals
    
    # Fallback
    if best_edges is None:
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
