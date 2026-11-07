import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 14
    m = 35
    max_val = 10000

    all_pairs = list(itertools.combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def calc_score(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    def get_edge_features(vals, edge):
        i, j = edge
        return vals[i] + vals[j], vals[i] * vals[j]

    def greedy_fill(vals, target_sums, target_prods, min_edges_needed, max_edges_to_add):
        """
        Fill edges greedily to reach min_edges_needed total.
        target_sums and target_prods are sets of allowed sums/prods.
        Returns list of edges added.
        """
        current_edges = []
        seen_sums = set()
        seen_prods = set()
        
        # Precompute all pairs with their features
        pair_info = []
        for idx, (i, j) in enumerate(all_pairs):
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            # Only consider pairs that fit within our target sets
            if s in target_sums and p in target_prods:
                pair_info.append((idx, i, j, s, p))
        
        # Sort by frequency (how many other pairs share the same sum/prod)
        sum_freq = Counter()
        prod_freq = Counter()
        for idx, i, j, s, p in pair_info:
            sum_freq[s] += 1
            prod_freq[p] += 1
        
        # Score each candidate: prefer high frequency (reuses sum/prod)
        def candidate_score(idx, i, j, s, p):
            new_s = 1 if s not in seen_sums else 0
            new_p = 1 if p not in seen_prods else 0
            # Prefer edges that don't add new sums/prods
            cost = new_s + new_p
            # Tie-break: prefer higher frequency
            freq = sum_freq[s] + prod_freq[p]
            return (cost, -freq)
        
        pair_info.sort(key=lambda x: candidate_score(*x))
        
        for idx, i, j, s, p in pair_info:
            if len(current_edges) >= max_edges_to_add:
                break
            if len(current_edges) >= min_edges_needed:
                break
            current_edges.append([i, j])
            seen_sums.add(s)
            seen_prods.add(p)
        
        return current_edges

    def backtracking_search(vals, target_sums, target_prods, max_depth=35):
        """
        Try to find exactly m edges using only sums in target_sums and prods in target_prods.
        Uses backtracking with pruning.
        """
        # Precompute valid edges
        valid_edges = []
        for idx, (i, j) in enumerate(all_pairs):
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            if s in target_sums and p in target_prods:
                valid_edges.append((idx, i, j, s, p))
        
        if len(valid_edges) < m:
            return None
        
        # Sort by degree (how many other valid edges share sum or prod)
        sum_count = Counter(s for _, _, _, s, _ in valid_edges)
        prod_count = Counter(p for _, _, _, _, p in valid_edges)
        
        def edge_degree(idx, i, j, s, p):
            return sum_count[s] + prod_count[p]
        
        valid_edges.sort(key=lambda x: -edge_degree(*x))
        
        # Backtracking
        selected = []
        used_sums = set()
        used_prods = set()
        
        def backtrack(start_idx, remaining):
            if remaining == 0:
                return True
            if start_idx >= len(valid_edges):
                return False
            
            # Pruning: not enough edges left
            if len(valid_edges) - start_idx < remaining:
                return False
            
            for idx in range(start_idx, len(valid_edges)):
                eidx, i, j, s, p = valid_edges[idx]
                
                # Check if this edge fits
                if s not in used_sums and len(used_sums) == len(target_sums):
                    continue
                if p not in used_prods and len(used_prods) == len(target_prods):
                    continue
                
                # Try selecting this edge
                selected.append([i, j])
                used_sums.add(s)
                used_prods.add(p)
                
                if backtrack(idx + 1, remaining - 1):
                    return True
                
                selected.pop()
                used_sums.discard(s)
                used_prods.discard(p)
            
            return False
        
        if backtrack(0, m):
            return selected
        return None

    def local_search(vals, edges, max_iters=200):
        """Improve edges via swaps to reduce max(distinct_sums, distinct_prods)."""
        current_edges = [e[:] for e in edges]
        current_score = calc_score(vals, current_edges)
        selected_keys = set(frozenset(e) for e in current_edges)
        non_selected = [list(p) for p in all_pairs if frozenset(p) not in selected_keys]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            random.shuffle(non_selected)
            
            for new_edge in non_selected:
                new_key = frozenset(new_edge)
                if new_key in selected_keys:
                    continue
                
                for k in range(m):
                    old_key = frozenset(current_edges[k])
                    if old_key == new_key:
                        continue
                    
                    temp = current_edges[:]
                    temp[k] = list(new_edge)
                    
                    if len(set(frozenset(e) for e in temp)) != m:
                        continue
                    
                    new_score = calc_score(vals, temp)
                    if new_score < current_score:
                        current_edges = temp
                        current_score = new_score
                        selected_keys = set(frozenset(e) for e in current_edges)
                        non_selected = [list(p) for p in all_pairs if frozenset(p) not in selected_keys]
                        improved = True
                        break
                if improved:
                    break
            iters += 1
        
        return current_edges, current_score

    best_score = float('inf')
    best_values = None
    best_edges = None

    # Generate candidate value sets
    value_sets = []
    
    # Arithmetic progressions
    for start in range(1, 20):
        for step in range(1, 10):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val:
                value_sets.append(vals)
    
    # Small random sets
    random.seed(42)
    for _ in range(30):
        vals = sorted(random.sample(range(1, 100), n))
        value_sets.append(vals)
    
    # Powers of 2 (sparse products)
    vals = [2**i for i in range(n)]
    if max(vals) <= max_val:
        value_sets.append(vals)
    
    # Mix of small and large
    for _ in range(20):
        small = random.sample(range(1, 20), 7)
        large = random.sample(range(100, 500), 7)
        vals = sorted(small + large)
        value_sets.append(vals)
    
    # Process value sets
    for vals in value_sets:
        if time.time() > deadline - 10:
            break
        
        # Try different target K values
        for K in range(5, 20):
            if time.time() > deadline - 10:
                break
            
            # Get all possible sums and products
            all_sums = set()
            all_prods = set()
            for i, j in all_pairs:
                all_sums.add(vals[i] + vals[j])
                all_prods.add(vals[i] * vals[j])
            
            # Try to find K sums and K prods that allow m edges
            # Use greedy approach: pick most frequent sums/prods
            sum_freq = Counter()
            prod_freq = Counter()
            for i, j in all_pairs:
                sum_freq[vals[i] + vals[j]] += 1
                prod_freq[vals[i] * vals[j]] += 1
            
            # Sort by frequency
            top_sums = [s for s, _ in sum_freq.most_common(K)]
            top_prods = [p for p, _ in prod_freq.most_common(K)]
            
            target_sums = set(top_sums)
            target_prods = set(top_prods)
            
            # Try backtracking
            edges = backtracking_search(vals, target_sums, target_prods)
            
            if edges is None:
                # Try greedy fill
                edges = greedy_fill(vals, target_sums, target_prods, m, m)
            
            if edges and len(edges) == m:
                score = calc_score(vals, edges)
                
                # Local search to improve
                if time.time() < deadline - 15:
                    edges, score = local_search(vals, edges, max_iters=50)
                
                if score < best_score:
                    best_score = score
                    best_values = vals[:]
                    best_edges = [e[:] for e in edges]
            
            # Also try with slightly larger K
            for K2 in [K+1, K+2]:
                if K2 > 25:
                    continue
                top_sums2 = [s for s, _ in sum_freq.most_common(K2)]
                top_prods2 = [p for p, _ in prod_freq.most_common(K2)]
                
                target_sums2 = set(top_sums2)
                target_prods2 = set(top_prods2)
                
                edges2 = backtracking_search(vals, target_sums2, target_prods2)
                
                if edges2 is None:
                    edges2 = greedy_fill(vals, target_sums2, target_prods2, m, m)
                
                if edges2 and len(edges2) == m:
                    score2 = calc_score(vals, edges2)
                    
                    if time.time() < deadline - 15:
                        edges2, score2 = local_search(vals, edges2, max_iters=30)
                    
                    if score2 < best_score:
                        best_score = score2
                        best_values = vals[:]
                        best_edges = [e[:] for e in edges2]

    # Fallback: if no good solution found, use simple approach
    if best_values is None:
        best_values = list(range(1, n + 1))
        # Simple greedy
        edges = []
        seen_sums = set()
        seen_prods = set()
        pair_data = []
        for i, j in all_pairs:
            s = best_values[i] + best_values[j]
            p = best_values[i] * best_values[j]
            pair_data.append((s, p, i, j))
        
        sum_freq = Counter(s for s, p, i, j in pair_data)
        prod_freq = Counter(p for s, p, i, j in pair_data)
        
        pair_data.sort(key=lambda x: -(sum_freq[x[0]] + prod_freq[x[1]]))
        
        for s, p, i, j in pair_data:
            if len(edges) >= m:
                break
            edges.append([i, j])
            seen_sums.add(s)
            seen_prods.add(p)
        
        best_edges = edges
        best_score = calc_score(best_values, best_edges)

    # Final type conversion
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]

    return {'values': best_values, 'edges': best_edges}
