import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin for 90s limit
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    best_result = None
    best_score = float('inf')
    
    # --- Phase 1: Generate diverse candidate value sets ---
    # Strategy: values with many shared factors produce fewer distinct products
    # and values that are close together produce fewer distinct sums.
    
    candidate_value_sets = []
    
    # Powers of 2 (products are powers of 2, sums vary)
    candidate_value_sets.append([2**i for i in range(14)])
    
    # Small composites with overlapping factors
    candidate_value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    candidate_value_sets.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32])
    candidate_value_sets.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    candidate_value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72])
    
    # Geometric-like sequences with small ratio
    for ratio_num, ratio_den in [(3, 2), (4, 3), (5, 4)]:
        vals = []
        v = 1
        for i in range(n):
            vals.append(v)
            v = v * ratio_num // ratio_den + 1
            if v > 10000:
                break
        if len(vals) == n and len(set(vals)) == n and all(1 <= x <= 10000 for x in vals):
            candidate_value_sets.append(vals)
    
    # Random composites: pick from a pool of numbers with many divisors
    random.seed(42)
    for trial in range(30):
        pool = [x for x in range(1, 200) if sum(1 for d in range(1, int(x**0.5)+1) if x % d == 0) >= 4]
        if len(pool) >= n:
            vals = random.sample(pool, n)
            if len(set(vals)) == n:
                candidate_value_sets.append(sorted(vals))
    
    # --- Phase 2: For each value set, run simulated annealing on edges ---
    
    def evaluate_edges(selected_set, edge_sums, edge_prods):
        """Compute distinct sums and products for a set of edge indices."""
        sum_count = {}
        prod_count = {}
        for idx in selected_set:
            s = edge_sums[idx]
            p = edge_prods[idx]
            sum_count[s] = sum_count.get(s, 0) + 1
            prod_count[p] = prod_count.get(p, 0) + 1
        return len(sum_count), len(prod_count)
    
    def score_edges(selected_set, edge_sums, edge_prods):
        ds, dp = evaluate_edges(selected_set, edge_sums, edge_prods)
        return max(ds, dp)
    
    for values in candidate_value_sets:
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # Greedy initialization
        selected = []
        sum_count = {}
        prod_count = {}
        distinct_sums = 0
        distinct_prods = 0
        
        for _ in range(m):
            best_idx = -1
            best_new_max = float('inf')
            best_new_ds = 0
            best_new_dp = 0
            
            for idx in range(num_all_edges):
                if idx in selected:
                    continue
                
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                new_ds = distinct_sums + (1 if s not in sum_count else 0)
                new_dp = distinct_prods + (1 if p not in prod_count else 0)
                new_max = max(new_ds, new_dp)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_idx = idx
                    best_new_ds = new_ds
                    best_new_dp = new_dp
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            s = edge_sums[best_idx]
            p = edge_prods[best_idx]
            sum_count[s] = sum_count.get(s, 0) + 1
            prod_count[p] = prod_count.get(p, 0) + 1
            distinct_sums = best_new_ds
            distinct_prods = best_new_dp
        
        current_score = max(distinct_sums, distinct_prods)
        
        # Simulated annealing
        if time.time() < deadline - 10:
            selected_set = set(selected)
            T = 10.0
            T_min = 0.01
            cooling = 0.995
            max_iter = 50000
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                if T < T_min:
                    break
                
                # Pick a random edge to remove and a random edge to add
                idx_out = random.choice(list(selected_set))
                idx_in = random.randint(0, num_all_edges - 1)
                while idx_in in selected_set:
                    idx_in = random.randint(0, num_all_edges - 1)
                
                # Compute new score
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]
                s_in = edge_sums[idx_in]
                p_in = edge_prods[idx_in]
                
                # Temporarily remove to compute new distinct counts
                sum_count[s_out] -= 1
                prod_count[p_out] -= 1
                ds_after = distinct_sums - (1 if sum_count[s_out] == 0 else 0)
                dp_after = distinct_prods - (1 if prod_count[p_out] == 0 else 0)
                
                new_ds = ds_after + (1 if sum_count.get(s_in, 0) == 0 else 0)
                new_dp = dp_after + (1 if prod_count.get(p_in, 0) == 0 else 0)
                new_score = max(new_ds, new_dp)
                
                delta = new_score - current_score
                
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < pow(2.718281828, -delta / T):
                        accept = True
                
                if accept:
                    # Apply swap
                    if sum_count[s_out] == 0:
                        del sum_count[s_out]
                        distinct_sums = ds_after
                    if prod_count[p_out] == 0:
                        del prod_count[p_out]
                        distinct_prods = dp_after
                    else:
                        distinct_sums = ds_after
                        distinct_prods = dp_after
                    
                    if s_in not in sum_count:
                        distinct_sums += 1
                    sum_count[s_in] = sum_count.get(s_in, 0) + 1
                    
                    if p_in not in prod_count:
                        distinct_prods += 1
                    prod_count[p_in] = prod_count.get(p_in, 0) + 1
                    
                    selected_set.remove(idx_out)
                    selected_set.add(idx_in)
                    current_score = max(distinct_sums, distinct_prods)
                else:
                    # Reject: restore
                    sum_count[s_out] += 1
                    prod_count[p_out] += 1
                
                T *= cooling
        
        if current_score < best_score:
            best_score = current_score
            best_result = {
                'values': list(values),
                'edges': [[all_edges[i][0], all_edges[i][1]] for i in selected_set]
            }
        
        if time.time() > deadline - 5:
            break
    
    if best_result is None:
        values = list(range(1, n + 1))
        best_result = {
            'values': values,
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    # Ensure correct types
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
