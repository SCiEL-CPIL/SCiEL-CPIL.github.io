"""Direct constructive approach for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        return ns, np_, max(ns, np_)
    
    def build_edge_map(values):
        """Map each (sum, product) to the list of edges that produce it."""
        edge_map = {}
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            key = (s, p)
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append((u, v))
        return edge_map
    
    def greedy_select_edges(edge_map, rng, target_m):
        """Greedily select edges to cover target_m edges, minimizing distinct (sum,product) groups."""
        groups = list(edge_map.items())
        rng.shuffle(groups)
        groups.sort(key=lambda x: -len(x[1]))
        
        selected_edges = []
        selected_set = set()
        
        for key, edges in groups:
            if len(selected_edges) >= target_m:
                break
            for e in edges:
                if len(selected_edges) >= target_m:
                    break
                if e not in selected_set:
                    selected_edges.append(e)
                    selected_set.add(e)
        
        if len(selected_edges) < target_m:
            for e in all_edges:
                if len(selected_edges) >= target_m:
                    break
                if e not in selected_set:
                    selected_edges.append(e)
                    selected_set.add(e)
        
        return selected_edges
    
    def local_search_edges(values, edges, rng, time_limit):
        """Local search on edges: prefer edges that share existing sums/products."""
        edges_set = set(tuple(e) for e in edges)
        _, _, current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(500):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                ns, np_, new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return edges, current_score
    
    def local_search_values(values, edges, rng, time_limit):
        """Local search on values: perturb one value at a time to reduce distinct sums/prods."""
        _, _, current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(80):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                
                candidates = set()
                for delta in range(-20, 21):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        candidates.add(v)
                for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]:
                    for mult in range(1, min(80, max_val // base + 1)):
                        candidates.add(base * mult)
                for base in [2, 3, 4, 5]:
                    v = 1
                    while v <= max_val:
                        candidates.add(v)
                        v *= base
                for _ in range(25):
                    candidates.add(rng.randint(1, max_val))
                
                best_new_val = None
                best_new_score = current_score
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in values:
                        continue
                    values[idx] = new_val
                    ns, np_, new_score = evaluate(values, edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    else:
                        values[idx] = old_val
                        break
                    values[idx] = old_val
                
                if best_new_val is not None:
                    values[idx] = best_new_val
                    current_score = best_new_score
                    improved = True
        
        return values, current_score
    
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Generate diverse value strategies with structural properties
    value_strategies = []
    
    # Arithmetic progressions
    for start in [1, 2, 3, 4, 5]:
        for step in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
            vals = [start + i * step for i in range(n)]
            if all(1 <= v <= max_val for v in vals) and len(set(vals)) == n:
                value_strategies.append(vals)
    
    # Geometric progressions
    for base in [2, 3]:
        for start in [1, 2, 3]:
            vals = [start * (base ** i) for i in range(n)]
            if all(1 <= v <= max_val for v in vals) and len(set(vals)) == n:
                value_strategies.append(vals)
    
    # Multiples of small numbers
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        vals = [base * (i + 1) for i in range(n)]
        if all(1 <= v <= max_val for v in vals) and len(set(vals)) == n:
            value_strategies.append(vals)
    
    # Symmetric pairs: (a_i, S - a_i)
    for S in [50, 60, 70, 80, 100, 120, 150, 200]:
        for spread in [1, 2, 3, 5, 7, 10]:
            vals = []
            for i in range(n // 2):
                a = S // 2 - spread * (n // 2 - 1 - i)
                b = S - a
                if a >= 1 and b >= 1 and a != b:
                    vals.extend([a, b])
            if len(vals) == n and len(set(vals)) == n and all(1 <= v <= max_val for v in vals):
                value_strategies.append(vals)
    
    # Small consecutive values
    value_strategies.append(list(range(1, n + 1)))
    value_strategies.append(list(range(2, n + 2)))
    
    # Mixed: some small, some large
    for small_count in [3, 4, 5, 6, 7]:
        for large_start in [50, 100, 200]:
            smalls = list(range(1, small_count + 1))
            larges = [large_start + i * 10 for i in range(n - small_count)]
            vals = smalls + larges
            if len(set(vals)) == n and all(1 <= v <= max_val for v in vals):
                value_strategies.append(vals)
    
    # Deduplicate
    seen = set()
    unique_strategies = []
    for vs in value_strategies:
        key = tuple(sorted(vs))
        if key not in seen:
            seen.add(key)
            unique_strategies.append(vs)
    value_strategies = unique_strategies
    
    # Phase 1: For each value strategy, use greedy edge selection + local search
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 10.0:
            break
        
        edge_map = build_edge_map(init_values)
        edges = greedy_select_edges(edge_map, rng, m)
        _, _, score = evaluate(init_values, edges)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': [list(e) for e in edges]}
        
        # Local search on edges
        if (time.time() - start_time) < deadline - 8.0:
            edges_ls = [list(e) for e in edges]
            edges_ls, score_e = local_search_edges(init_values, edges_ls, rng, deadline - 8.0)
            if score_e < best_overall_score:
                best_overall_score = score_e
                best_result = {'values': list(init_values), 'edges': [list(e) for e in edges_ls]}
        
        # Local search on values
        if (time.time() - start_time) < deadline - 5.0:
            values_ls = list(best_result['values'])
            edges_ls = [list(e) for e in best_result['edges']]
            values_ls, score_v = local_search_values(values_ls, edges_ls, rng, deadline - 5.0)
            if score_v < best_overall_score:
                best_overall_score = score_v
                best_result = {'values': list(values_ls), 'edges': [list(e) for e in edges_ls]}
    
    # Phase 2: Aggressive refinement
    if best_result is not None:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        for iteration in range(50):
            if (time.time() - start_time) > deadline - 2.0:
                break
            
            values, score_v = local_search_values(values, edges, rng, deadline - 2.0)
            if score_v < best_overall_score:
                best_overall_score = score_v
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
            
            edges, score_e = local_search_edges(values, edges, rng, deadline - 2.0)
            if score_e < best_overall_score:
                best_overall_score = score_e
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
