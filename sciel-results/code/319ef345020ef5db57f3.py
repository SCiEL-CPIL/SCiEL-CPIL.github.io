import time
import random
from collections import Counter
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 14
    m = 35
    maximum = 10000
    
    # Precompute all pairs
    all_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.append((i, j))
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    def evaluate_solution(values, edge_indices):
        """Compute raw value for given values and edge indices."""
        sums = set()
        prods = set()
        for idx in edge_indices:
            i, j = all_pairs[idx]
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    def constraint_based_edge_selection(values, target_k):
        """
        Constraint-based approach: find edges that can be colored with at most K
        distinct sums and K distinct products. Use backtracking with pruning.
        
        Returns (edge_indices, actual_k) or None if infeasible.
        """
        pair_sum = [values[i] + values[j] for i, j in all_pairs]
        pair_prod = [values[i] * values[j] for i, j in all_pairs]
        
        # Group edges by their (sum, prod) pair
        edge_groups = {}
        for idx in range(len(all_pairs)):
            key = (pair_sum[idx], pair_prod[idx])
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(idx)
        
        # Sort groups by size (larger groups first - more edges per color)
        sorted_groups = sorted(edge_groups.items(), key=lambda x: -len(x[1]))
        
        # Greedy: select edges from groups to minimize distinct sum/prod count
        # While ensuring we can pick exactly m edges
        used_sums = set()
        used_prods = set()
        selected_edges = []
        
        # First pass: take all edges from the largest groups
        for key, edges in sorted_groups:
            if len(selected_edges) >= m:
                break
            s, p = key
            # Check if adding this group's edges would exceed target_k
            new_sums = len(used_sums | {s})
            new_prods = len(used_prods | {p})
            if max(new_sums, new_prods) <= target_k:
                for e in edges:
                    if len(selected_edges) >= m:
                        break
                    selected_edges.append(e)
                used_sums.add(s)
                used_prods.add(p)
        
        # If we have enough edges, we're done
        if len(selected_edges) >= m:
            actual_k = max(len(used_sums), len(used_prods))
            return selected_edges[:m], actual_k
        
        # Second pass: add edges from remaining groups, preferring those
        # that reuse existing sums or products
        remaining_groups = [g for g in sorted_groups if len(g[1]) > 0 and g[1][0] not in selected_edges]
        
        # Score each edge by how much it reuses colors
        def edge_score(idx):
            s = pair_sum[idx]
            p = pair_prod[idx]
            score = 0
            if s in used_sums:
                score += 10
            if p in used_prods:
                score += 10
            # Prefer edges whose sum/prod are already used
            return -score
        
        remaining_edges = [idx for idx in range(len(all_pairs)) if idx not in selected_edges]
        remaining_edges.sort(key=lambda idx: edge_score(idx))
        
        for idx in remaining_edges:
            if len(selected_edges) >= m:
                break
            s = pair_sum[idx]
            p = pair_prod[idx]
            new_sums = len(used_sums | {s})
            new_prods = len(used_prods | {p})
            if max(new_sums, new_prods) <= target_k:
                selected_edges.append(idx)
                used_sums.add(s)
                used_prods.add(p)
        
        if len(selected_edges) >= m:
            actual_k = max(len(used_sums), len(used_prods))
            return selected_edges[:m], actual_k
        
        return None, float('inf')
    
    def local_search_edges(values, initial_edges, max_iters=30000, seed=0):
        """Local search with 1-swap and 2-swap moves to minimize raw value."""
        rng = random.Random(seed)
        edge_set = set(initial_edges)
        
        pair_sum = [values[i] + values[j] for i, j in all_pairs]
        pair_prod = [values[i] * values[j] for i, j in all_pairs]
        
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in edge_set:
            sum_counts[pair_sum[idx]] += 1
            prod_counts[pair_prod[idx]] += 1
        
        def get_raw():
            return max(len(sum_counts), len(prod_counts))
        
        current_raw = get_raw()
        best_raw_so_far = current_raw
        best_edge_set = set(edge_set)
        
        T = 3.0
        T_min = 0.05
        cooling = 0.99995
        
        unselected = set(range(len(all_pairs))) - edge_set
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if not edge_set or not unselected:
                continue
            
            if rng.random() < 0.75:
                # 1-swap
                remove_idx = rng.choice(list(edge_set))
                candidates = rng.sample(list(unselected), min(8, len(unselected)))
                
                best_candidate = None
                best_new_raw = current_raw
                
                for add_idx in candidates:
                    if add_idx == remove_idx:
                        continue
                    
                    rs, rp = pair_sum[remove_idx], pair_prod[remove_idx]
                    as_, ap = pair_sum[add_idx], pair_prod[add_idx]
                    
                    sum_counts[rs] -= 1
                    if sum_counts[rs] == 0: del sum_counts[rs]
                    sum_counts[as_] += 1
                    
                    prod_counts[rp] -= 1
                    if prod_counts[rp] == 0: del prod_counts[rp]
                    prod_counts[ap] += 1
                    
                    new_raw = get_raw()
                    
                    if new_raw < best_new_raw:
                        best_new_raw = new_raw
                        best_candidate = add_idx
                    
                    sum_counts[rs] += 1
                    if sum_counts[as_] == 1: del sum_counts[as_]
                    else: sum_counts[as_] -= 1
                    
                    prod_counts[rp] += 1
                    if prod_counts[ap] == 1: del prod_counts[ap]
                    else: prod_counts[ap] -= 1
                
                if best_candidate is None:
                    continue
                
                add_idx = best_candidate
                rs, rp = pair_sum[remove_idx], pair_prod[remove_idx]
                as_, ap = pair_sum[add_idx], pair_prod[add_idx]
                
                sum_counts[rs] -= 1
                if sum_counts[rs] == 0: del sum_counts[rs]
                sum_counts[as_] += 1
                
                prod_counts[rp] -= 1
                if prod_counts[rp] == 0: del prod_counts[rp]
                prod_counts[ap] += 1
                
                new_raw = best_new_raw
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    edge_set.remove(remove_idx)
                    edge_set.add(add_idx)
                    unselected.add(remove_idx)
                    unselected.remove(add_idx)
                    
                    current_raw = new_raw
                    if current_raw < best_raw_so_far:
                        best_raw_so_far = current_raw
                        best_edge_set = set(edge_set)
                else:
                    sum_counts[rs] += 1
                    if sum_counts[as_] == 1: del sum_counts[as_]
                    else: sum_counts[as_] -= 1
                    
                    prod_counts[rp] += 1
                    if prod_counts[ap] == 1: del prod_counts[ap]
                    else: prod_counts[ap] -= 1
            else:
                # 2-swap
                if len(edge_set) < 2 or len(unselected) < 2:
                    continue
                
                rem1, rem2 = rng.sample(list(edge_set), 2)
                add1, add2 = rng.sample(list(unselected), 2)
                
                if rem1 in (add1, add2) or rem2 in (add1, add2):
                    continue
                
                s1, p1 = pair_sum[rem1], pair_prod[rem1]
                s2, p2 = pair_sum[rem2], pair_prod[rem2]
                
                sum_counts[s1] -= 1
                if sum_counts[s1] == 0: del sum_counts[s1]
                prod_counts[p1] -= 1
                if prod_counts[p1] == 0: del prod_counts[p1]
                
                sum_counts[s2] -= 1
                if sum_counts[s2] == 0: del sum_counts[s2]
                prod_counts[p2] -= 1
                if prod_counts[p2] == 0: del prod_counts[p2]
                
                sa1, pa1 = pair_sum[add1], pair_prod[add1]
                sa2, pa2 = pair_sum[add2], pair_prod[add2]
                
                sum_counts[sa1] += 1
                prod_counts[pa1] += 1
                sum_counts[sa2] += 1
                prod_counts[pa2] += 1
                
                new_raw = get_raw()
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    edge_set.discard(rem1)
                    edge_set.discard(rem2)
                    edge_set.add(add1)
                    edge_set.add(add2)
                    
                    unselected.add(rem1)
                    unselected.add(rem2)
                    unselected.discard(add1)
                    unselected.discard(add2)
                    
                    current_raw = new_raw
                    if current_raw < best_raw_so_far:
                        best_raw_so_far = current_raw
                        best_edge_set = set(edge_set)
                else:
                    sum_counts[sa1] -= 1
                    if sum_counts[sa1] == 0: del sum_counts[sa1]
                    prod_counts[pa1] -= 1
                    if prod_counts[pa1] == 0: del prod_counts[pa1]
                    
                    sum_counts[sa2] -= 1
                    if sum_counts[sa2] == 0: del sum_counts[sa2]
                    prod_counts[pa2] -= 1
                    if prod_counts[pa2] == 0: del prod_counts[pa2]
                    
                    sum_counts[s1] += 1
                    prod_counts[p1] += 1
                    sum_counts[s2] += 1
                    prod_counts[p2] += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return list(best_edge_set), best_raw_so_far
    
    def generate_structured_values():
        """Generate diverse structured value sets."""
        sets = []
        
        # Small factor-rich numbers
        base_sets = [
            [2,3,4,5,6,8,9,10,12,15,16,18,20,24],
            [1,2,3,4,5,6,7,8,9,10,12,15,18,20],
            [2,4,6,8,10,12,14,16,18,20,24,30,36,40],
            [3,4,5,6,8,9,10,12,15,16,18,20,24,30],
            [1,2,3,4,6,8,9,12,15,18,20,24,30,36],
            [2,4,6,9,10,12,15,16,18,20,24,27,30,36],
            [1,2,4,6,8,9,10,12,15,16,18,20,24,30],
            [2,3,6,8,9,10,12,15,16,18,20,24,27,30],
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 10000, 5000, 2500, 1250, 625],
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [2, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 9000, 9500],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
            [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 9999],
        ]
        sets.extend(base_sets)
        
        # Arithmetic progressions
        for a in range(1, 10):
            for d in range(1, 5):
                vals = [a + i*d for i in range(n)]
                if all(1 <= v <= maximum for v in vals) and len(set(vals)) == n:
                    sets.append(vals)
        
        # Geometric-like sequences
        for base in [2, 3, 4]:
            for start in [1, 2]:
                vals = []
                v = start
                for _ in range(n):
                    vals.append(v)
                    v = v * base // 2 + 1  # approximate growth
                if all(1 <= x <= maximum for x in vals) and len(set(vals)) == n:
                    sets.append(vals)
        
        # Random small sets
        rng = random.Random(42)
        for _ in range(100):
            vals = sorted(rng.sample(range(1, 80), n))
            sets.append(vals)
        
        # Random sets with structure
        for _ in range(50):
            # Mix of small and slightly larger
            small = rng.sample(range(1, 30), 7)
            medium = rng.sample(range(30, 100), 5)
            large = rng.sample(range(100, 500), 2)
            vals = sorted(small + medium + large)
            if len(set(vals)) == n and all(1 <= v <= maximum for v in vals):
                sets.append(vals)
        
        return sets
    
    # Generate candidate value sets
    candidate_values = generate_structured_values()
    
    # Try each candidate value set with constraint-based + local search
    for vi, values in enumerate(candidate_values):
        if time.time() > deadline - 5:
            break
        
        if len(set(values)) != n or any(v < 1 or v > maximum for v in values):
            continue
        
        # Try constraint-based selection with different target K values
        for target_k in [12, 13, 14, 15, 16, 17, 18]:
            if time.time() > deadline - 5:
                break
            
            cb_edges, cb_raw = constraint_based_edge_selection(values, target_k)
            
            if cb_edges is not None and cb_raw < best_raw:
                best_raw = cb_raw
                best_values = list(values)
                best_edges = [list(all_pairs[idx]) for idx in cb_edges]
                
                if best_raw <= 12:
                    return {'values': best_values, 'edges': best_edges}
            
            # Refine with local search
            if cb_edges is not None:
                ls_edges, ls_raw = local_search_edges(values, cb_edges, max_iters=15000, seed=vi*100+target_k)
                
                if ls_raw < best_raw:
                    best_raw = ls_raw
                    best_values = list(values)
                    best_edges = [list(all_pairs[idx]) for idx in ls_edges]
                    
                    if best_raw <= 12:
                        return {'values': best_values, 'edges': best_edges}
        
        # Also try direct local search from random initial edges
        if time.time() > deadline - 10:
            break
        
        rng_init = random.Random(vi * 7 + 1)
        random_edges = rng_init.sample(range(len(all_pairs)), m)
        ls_edges, ls_raw = local_search_edges(values, random_edges, max_iters=10000, seed=vi*50+1)
        
        if ls_raw < best_raw:
            best_raw = ls_raw
            best_values = list(values)
            best_edges = [list(all_pairs[idx]) for idx in ls_edges]
            
            if best_raw <= 12:
                return {'values': best_values, 'edges': best_edges}
    
    # Fallback
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges = [list(e) for e in itertools.combinations(range(n), 2)[:m]]
    
    return {'values': best_values, 'edges': best_edges}
