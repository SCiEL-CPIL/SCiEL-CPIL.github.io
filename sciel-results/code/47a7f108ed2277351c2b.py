"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def _evaluate(values, edges):
    """Return (num_distinct_sums, num_distinct_products, max_of_those)."""
    sums = set()
    prods = set()
    for (i, j) in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    ns = len(sums)
    np_ = len(prods)
    return ns, np_, max(ns, np_)

def _greedy_pick_edges(values, m, deadline):
    """Greedy: pick edges that minimize max(distinct_sums, distinct_products)."""
    n = len(values)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_sum = {}
    edge_prod = {}
    for (i, j) in all_edges:
        edge_sum[(i, j)] = values[i] + values[j]
        edge_prod[(i, j)] = values[i] * values[j]
    
    selected = []
    selected_set = set()
    sums_set = set()
    prods_set = set()
    
    for _ in range(m):
        if time.time() > deadline:
            break
        best_edge = None
        best_score = float('inf')
        for e in all_edges:
            if e in selected_set:
                continue
            s = edge_sum[e]
            p = edge_prod[e]
            # Cost of adding this edge
            add_s = 1 if s not in sums_set else 0
            add_p = 1 if p not in prods_set else 0
            new_sums = len(sums_set) + add_s
            new_prods = len(prods_set) + add_p
            score = max(new_sums, new_prods)
            if score < best_score:
                best_score = score
                best_edge = e
        if best_edge is None:
            break
        selected.append(best_edge)
        selected_set.add(best_edge)
        sums_set.add(edge_sum[best_edge])
        prods_set.add(edge_prod[best_edge])
    
    return selected

def _local_search_edges(values, edges, m, deadline, max_iters=3000):
    """Local search: swap edges to improve score, with incremental tracking."""
    n = len(values)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_sum = {}
    edge_prod = {}
    for (i, j) in all_edges:
        edge_sum[(i, j)] = values[i] + values[j]
        edge_prod[(i, j)] = values[i] * values[j]
    
    selected = list(edges)
    selected_set = set(selected)
    
    # Build frequency maps
    from collections import Counter
    sum_freq = Counter()
    prod_freq = Counter()
    for e in selected:
        sum_freq[edge_sum[e]] += 1
        prod_freq[edge_prod[e]] += 1
    
    def current_score():
        ns = len(sum_freq)
        np_ = len(prod_freq)
        return max(ns, np_)
    
    current_score_val = current_score()
    best_score = current_score_val
    best_selected = list(selected)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(selected)):
            if time.time() > deadline:
                break
            e_out = selected[idx]
            s_out = edge_sum[e_out]
            p_out = edge_prod[e_out]
            
            # Remove e_out
            sum_freq[s_out] -= 1
            if sum_freq[s_out] == 0:
                del sum_freq[s_out]
            prod_freq[p_out] -= 1
            if prod_freq[p_out] == 0:
                del prod_freq[p_out]
            selected_set.discard(e_out)
            
            for e_in in all_edges:
                if e_in in selected_set:
                    continue
                s_in = edge_sum[e_in]
                p_in = edge_prod[e_in]
                
                # Add e_in
                sum_freq[s_in] = sum_freq.get(s_in, 0) + 1
                prod_freq[p_in] = prod_freq.get(p_in, 0) + 1
                selected_set.add(e_in)
                
                new_score = current_score()
                
                if new_score < current_score_val:
                    # Accept
                    selected[idx] = e_in
                    current_score_val = new_score
                    if current_score_val < best_score:
                        best_score = current_score_val
                        best_selected = list(selected)
                    improved = True
                    break
                else:
                    # Revert
                    sum_freq[s_in] -= 1
                    if sum_freq[s_in] == 0:
                        del sum_freq[s_in]
                    prod_freq[p_in] -= 1
                    if prod_freq[p_in] == 0:
                        del prod_freq[p_in]
                    selected_set.discard(e_in)
            
            # Restore e_out
            sum_freq[s_out] = sum_freq.get(s_out, 0) + 1
            prod_freq[p_out] = prod_freq.get(p_out, 0) + 1
            selected_set.add(e_out)
            
            if improved:
                break
        if not improved:
            break
    
    return best_selected

def construct():
    deadline = time.time() + 70
    n = 14
    m = 35
    
    best_result = None
    best_score = float('inf')
    
    value_configs = []
    
    # Arithmetic progressions: start + i*d for i in 0..13
    # max value = start + 13*d <= 10000
    for start in range(1, 50):
        max_d = (10000 - start) // 13
        for d in range(1, min(max_d + 1, 500)):
            vals = [start + i * d for i in range(n)]
            if len(set(vals)) == n and all(v <= 10000 for v in vals):
                value_configs.append(vals)
        if time.time() > deadline - 40:
            break
    
    # Add specific good configs
    value_configs.append(list(range(1, n + 1)))  # 1-14
    value_configs.append([2 ** i for i in range(n)])  # 1,2,4,...,8192
    
    # Sample if too many
    random.seed(42)
    if len(value_configs) > 150:
        value_configs = random.sample(value_configs, 150)
    
    for vals in value_configs:
        if time.time() > deadline - 15:
            break
        if len(set(vals)) != n or any(v <= 0 or v > 10000 for v in vals):
            continue
        vals = [int(v) for v in vals]
        
        edges = _greedy_pick_edges(vals, m, deadline - 10)
        if len(edges) < m:
            continue
        
        edges = _local_search_edges(vals, edges, m, deadline - 5, max_iters=1500)
        
        ns, np_, score = _evaluate(vals, edges)
        if score < best_score:
            best_score = score
            best_result = {'values': vals, 'edges': [list(e) for e in edges]}
        
        if time.time() > deadline - 10:
            break
    
    if best_result is None:
        all_edges = list(itertools.combinations(range(n), 2))[:m]
        best_result = {'values': list(range(1, n + 1)), 'edges': [list(e) for e in all_edges]}
    
    result = {
        'values': [int(v) for v in best_result['values']],
        'edges': [[int(i), int(j)] for i, j in best_result['edges']]
    }
    return result
# EVOLVE-BLOCK-END
