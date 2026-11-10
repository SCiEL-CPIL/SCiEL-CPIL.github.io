import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 82.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    all_edges = list(combinations(range(n), 2))
    E = len(all_edges)
    
    def compute_score_and_counts(values, edge_indices):
        """Compute score and maintain counts for incremental updates."""
        sum_counts = {}
        prod_counts = {}
        for idx in edge_indices:
            u, v = all_edges[idx]
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        return max(distinct_sums, distinct_prods), sum_counts, prod_counts
    
    def try_values_search(values, time_budget):
        """Local search on edges for given values, using incremental tracking."""
        edge_sums = [0] * E
        edge_prods = [0] * E
        for idx, (u, v) in enumerate(all_edges):
            edge_sums[idx] = values[u] + values[v]
            edge_prods[idx] = values[u] * values[v]
        
        best_score = float('inf')
        best_edges = None
        
        num_restarts = 200
        for r in range(num_restarts):
            if time.time() > time_budget:
                break
            
            current_edges = set(random.sample(range(E), m))
            current_score, sum_counts, prod_counts = compute_score_and_counts(values, current_edges)
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
            
            # Simulated annealing with incremental updates
            T = 3.0
            steps = 500
            for s in range(steps):
                if time.time() > time_budget:
                    break
                
                e_out = random.choice(list(current_edges))
                e_in = random.randint(0, E - 1)
                if e_in in current_edges:
                    continue
                
                s_out = edge_sums[e_out]
                p_out = edge_prods[e_out]
                s_in = edge_sums[e_in]
                p_in = edge_prods[e_in]
                
                # Apply removal
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                # Apply addition
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                
                if new_score < current_score or random.random() < min(1.0, T / max(1, current_score)):
                    current_edges.discard(e_out)
                    current_edges.add(e_in)
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(current_edges)
                else:
                    # Revert
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        del sum_counts[s_in]
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        del prod_counts[p_in]
                    
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                
                T *= 0.995
            
            # Exhaustive single-swap local search with incremental tracking
            if time.time() > time_budget:
                break
            current_edges = set(best_edges)
            current_score, sum_counts, prod_counts = compute_score_and_counts(values, current_edges)
            
            improved = True
            while improved:
                improved = False
                for e_out in list(current_edges):
                    if time.time() > time_budget:
                        break
                    s_out = edge_sums[e_out]
                    p_out = edge_prods[e_out]
                    
                    for e_in in range(E):
                        if e_in in current_edges:
                            continue
                        s_in = edge_sums[e_in]
                        p_in = edge_prods[e_in]
                        
                        if s_out == s_in and p_out == p_in:
                            continue
                        
                        # Apply removal
                        sum_counts[s_out] -= 1
                        if sum_counts[s_out] == 0:
                            del sum_counts[s_out]
                        prod_counts[p_out] -= 1
                        if prod_counts[p_out] == 0:
                            del prod_counts[p_out]
                        
                        # Apply addition
                        sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                        prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                        
                        new_score = max(len(sum_counts), len(prod_counts))
                        
                        if new_score < current_score:
                            current_edges.discard(e_out)
                            current_edges.add(e_in)
                            current_score = new_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_edges = set(current_edges)
                            break
                        else:
                            # Revert
                            sum_counts[s_in] -= 1
                            if sum_counts[s_in] == 0:
                                del sum_counts[s_in]
                            prod_counts[p_in] -= 1
                            if prod_counts[p_in] == 0:
                                del prod_counts[p_in]
                            
                            sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                            prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                    
                    if time.time() > time_budget:
                        break
                
                if time.time() > time_budget:
                    break
        
        return best_score, best_edges
    
    # Phase 1: Try the base values 1..14
    base_values = list(range(1, n + 1))
    best_overall_score, best_overall_edges = try_values_search(base_values, deadline - 10.0)
    
    # Phase 2: Try alternative value sets
    if time.time() < deadline - 5.0:
        alternative_value_sets = [
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        ]
        
        for vals in alternative_value_sets:
            if time.time() > deadline - 5.0:
                break
            if len(set(vals)) != n or any(v < 1 or v > maximum for v in vals):
                continue
            
            score, edges = try_values_search(vals, deadline - 5.0)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_edges = edges
                base_values = vals
    
    # Final polish
    if best_overall_edges is not None and time.time() < deadline - 2.0:
        edge_sums = [0] * E
        edge_prods = [0] * E
        for idx, (u, v) in enumerate(all_edges):
            edge_sums[idx] = base_values[u] + base_values[v]
            edge_prods[idx] = base_values[u] * base_values[v]
        
        current_edges = set(best_overall_edges)
        current_score, sum_counts, prod_counts = compute_score_and_counts(base_values, current_edges)
        
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            improved = True
            while improved:
                improved = False
                for e_out in list(current_edges):
                    if time.time() > deadline - 2.0:
                        break
                    s_out = edge_sums[e_out]
                    p_out = edge_prods[e_out]
                    
                    for e_in in range(E):
                        if e_in in current_edges:
                            continue
                        s_in = edge_sums[e_in]
                        p_in = edge_prods[e_in]
                        
                        if s_out == s_in and p_out == p_in:
                            continue
                        
                        sum_counts[s_out] -= 1
                        if sum_counts[s_out] == 0:
                            del sum_counts[s_out]
                        prod_counts[p_out] -= 1
                        if prod_counts[p_out] == 0:
                            del prod_counts[p_out]
                        
                        sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                        prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                        
                        new_score = max(len(sum_counts), len(prod_counts))
                        
                        if new_score < current_score:
                            current_edges.discard(e_out)
                            current_edges.add(e_in)
                            current_score = new_score
                            improved = True
                            if current_score < best_overall_score:
                                best_overall_score = current_score
                                best_overall_edges = set(current_edges)
                            break
                        else:
                            sum_counts[s_in] -= 1
                            if sum_counts[s_in] == 0:
                                del sum_counts[s_in]
                            prod_counts[p_in] -= 1
                            if prod_counts[p_in] == 0:
                                del prod_counts[p_in]
                            
                            sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                            prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                    
                    if time.time() > deadline - 2.0:
                        break
                if time.time() > deadline - 2.0:
                    break
    
    if best_overall_edges is None:
        best_overall_edges = set(range(m))
        base_values = list(range(1, n + 1))
    
    edges_list = [[all_edges[idx][0], all_edges[idx][1]] for idx in sorted(best_overall_edges)]
    
    return {
        'values': [int(v) for v in base_values],
        'edges': edges_list
    }
