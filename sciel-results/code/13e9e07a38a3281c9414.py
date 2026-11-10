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
    
    # Focused candidate value sets with different structural properties
    candidates = [
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 40, 60],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 45, 60, 90],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 24],
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
        if time.time() > deadline - 10.0:
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
        
        # Precompute: for each edge, how many other edges share its sum or product
        edge_share_sum = [0] * num_edges_total
        edge_share_prod = [0] * num_edges_total
        edge_share_both = [0] * num_edges_total
        
        for idx in range(num_edges_total):
            s = edge_sums[idx]
            p = edge_prods[idx]
            for idx2 in range(num_edges_total):
                if idx2 == idx:
                    continue
                ss = edge_sums[idx2]
                pp = edge_prods[idx2]
                if ss == s:
                    edge_share_sum[idx] += 1
                if pp == p:
                    edge_share_prod[idx] += 1
                if ss == s and pp == p:
                    edge_share_both[idx] += 1
        
        # Greedy construction
        edge_scores = []
        for idx in range(num_edges_total):
            combined = edge_share_both[idx] * 10 + edge_share_sum[idx] + edge_share_prod[idx]
            edge_scores.append((combined, sum_freq[edge_sums[idx]], prod_freq[edge_prods[idx]], idx))
        edge_scores.sort(reverse=True)
        
        selected = []
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        distinct_sums = 0
        distinct_prods = 0
        
        for _, _, _, idx in edge_scores:
            if len(selected) >= m:
                break
            s = edge_sums[idx]
            p = edge_prods[idx]
            selected.append(idx)
            if s not in sum_counts:
                distinct_sums += 1
            sum_counts[s] += 1
            if p not in prod_counts:
                distinct_prods += 1
            prod_counts[p] += 1
        
        current_score = max(distinct_sums, distinct_prods)
        
        # Local search
        if time.time() < deadline - 8.0:
            improved = True
            max_iters = 300
            iter_count = 0
            
            while improved and iter_count < max_iters and time.time() < deadline - 8.0:
                improved = False
                iter_count += 1
                
                # Try random swaps
                for _ in range(50):
                    if time.time() > deadline - 8.0:
                        break
                    
                    out_pos = random.randint(0, len(selected) - 1)
                    idx_out = selected[out_pos]
                    selected_set = set(selected)
                    
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    
                    # Find best candidate to swap in
                    best_candidate = None
                    best_candidate_score = current_score + 1
                    
                    for idx in range(num_edges_total):
                        if idx in selected_set:
                            continue
                        s_in = edge_sums[idx]
                        p_in = edge_prods[idx]
                        
                        adds_sum = 1 if s_in not in sum_counts else 0
                        adds_prod = 1 if p_in not in prod_counts else 0
                        frees_sum = 1 if sum_counts[s_out] == 1 else 0
                        frees_prod = 1 if prod_counts[p_out] == 1 else 0
                        
                        new_ds = distinct_sums + adds_sum - frees_sum
                        new_dp = distinct_prods + adds_prod - frees_prod
                        new_score = max(new_ds, new_dp)
                        
                        if new_score < best_candidate_score:
                            best_candidate_score = new_score
                            best_candidate = idx
                    
                    if best_candidate is None:
                        continue
                    
                    idx_in = best_candidate
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    # Apply swap
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
                    else:
                        # Revert
                        sum_counts[s_in] -= 1
                        if sum_counts[s_in] == 0:
                            del sum_counts[s_in]
                        prod_counts[p_in] -= 1
                        if prod_counts[p_in] == 0:
                            del prod_counts[p_in]
                        
                        sum_counts[s_out] += 1
                        prod_counts[p_out] += 1
        
        # Strategy 2: Target palette approach
        if time.time() < deadline - 5.0:
            for target_k in range(5, 12):
                if time.time() > deadline - 5.0:
                    break
                
                top_sums = sorted(sum_freq.items(), key=lambda x: -x[1])[:target_k]
                top_sum_set = set(s for s, _ in top_sums)
                
                top_prods = sorted(prod_freq.items(), key=lambda x: -x[1])[:target_k]
                top_prod_set = set(p for p, _ in top_prods)
                
                good_edges = []
                for idx in range(num_edges_total):
                    if edge_sums[idx] in top_sum_set and edge_prods[idx] in top_prod_set:
                        good_edges.append(idx)
                
                if len(good_edges) >= m:
                    sel = good_edges[:m]
                    sc = defaultdict(int)
                    pc = defaultdict(int)
                    for idx in sel:
                        sc[edge_sums[idx]] += 1
                        pc[edge_prods[idx]] += 1
                    ds = len(sc)
                    dp = len(pc)
                    sc_score = max(ds, dp)
                    if sc_score < current_score:
                        current_score = sc_score
                        selected = list(sel)
        
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
