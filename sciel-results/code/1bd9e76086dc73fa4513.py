import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    # Precompute all edges
    all_edges = list(combinations(range(n), 2))
    E = len(all_edges)  # 91
    
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
        # Precompute sums and products for all edges
        edge_sums = [0] * E
        edge_prods = [0] * E
        for idx, (u, v) in enumerate(all_edges):
            edge_sums[idx] = values[u] + values[v]
            edge_prods[idx] = values[u] * values[v]
        
        best_score = float('inf')
        best_edges = None
        
        # Multiple restarts with simulated annealing + local search
        num_restarts = 300
        for r in range(num_restarts):
            if time.time() > time_budget:
                break
            
            # Random initial selection
            current_edges = set(random.sample(range(E), m))
            current_score, sum_counts, prod_counts = compute_score_and_counts(values, current_edges)
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
            
            # Simulated annealing with incremental updates
            T = 3.0
            steps = 400
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
                
                # Accept if better or with probability based on temperature
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
    
    # Generate candidate value sets
    random.seed(42)
    candidates = []
    
    # 1. Consecutive integers
    candidates.append(list(range(1, 15)))
    
    # 2. Powers of 2
    candidates.append([2**i for i in range(14)])
    
    # 3. Arithmetic progressions
    for start in [1, 2, 3]:
        for d in [1, 2, 3, 5]:
            vals = [start + i*d for i in range(14)]
            if max(vals) <= maximum:
                candidates.append(vals)
    
    # 4. Structured sets with many shared sums/products
    # Multiples of a base
    for base in [2, 3, 5, 10]:
        for start in range(1, 10):
            vals = [start * base + i * base for i in range(14)]
            if max(vals) <= maximum and len(set(vals)) == 14:
                candidates.append(vals)
    
    # 5. Random sets with structure
    for _ in range(30):
        base = random.randint(1, 10)
        multipliers = random.sample(range(1, 25), 14)
        vals = sorted([m * base for m in multipliers])
        if len(set(vals)) == 14 and max(vals) <= maximum:
            candidates.append(vals)
    
    # 6. Random distinct integers
    for _ in range(30):
        vals = random.sample(range(1, 100), 14)
        candidates.append(vals)
    
    # 7. Near-consecutive with one gap
    for gap_pos in range(1, 13):
        vals = list(range(1, 15))
        vals[gap_pos] += 1
        if len(set(vals)) == 14 and max(vals) <= maximum:
            candidates.append(vals)
    
    # Evaluate each candidate
    best_global_score = float('inf')
    best_global_vals = None
    best_global_edges = None
    
    # Allocate time proportionally: more time for earlier (likely better) candidates
    total_time = deadline - start_time - 15.0  # Reserve 15s for safety
    if total_time < 10:
        total_time = 10
    
    # Prioritize consecutive integers first
    candidates_sorted = [list(range(1, 15))] + [c for c in candidates if c != list(range(1, 15))]
    
    for idx, vals in enumerate(candidates_sorted):
        if time.time() > deadline - 15:
            break
        
        # Allocate time: first candidate gets 40%, rest share 60%
        if idx == 0:
            time_budget = deadline - 15 - total_time * 0.6
        else:
            time_budget = deadline - 15 - (total_time * 0.6 / max(1, len(candidates_sorted) - idx - 1))
        
        if time_budget < 1.0:
            continue
        
        score, edges = try_values_search(vals, time_budget)
        if score < best_global_score:
            best_global_score = score
            best_global_vals = vals[:]
            best_global_edges = edges.copy()
            
            # If we found a very good score, we can stop early
            if best_global_score <= 10:
                break
    
    # Fallback
    if best_global_vals is None:
        best_global_vals = list(range(1, 15))
        best_global_edges = set(random.sample(range(E), m))
        best_global_score, _, _ = compute_score_and_counts(best_global_vals, best_global_edges)
    
    # Construct final result
    final_values = [int(v) for v in best_global_vals]
    final_edges = []
    for idx in best_global_edges:
        i, j = all_edges[idx]
        final_edges.append([int(i), int(j)])
    
    return {'values': final_values, 'edges': final_edges}
