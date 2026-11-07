"""Direct constructive approach for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin for 90s limit
    
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
    
    def fast_edge_local_search(values, edges, rng, time_limit):
        """O(1) per-swap edge local search using incremental sum/product tracking."""
        # Build initial state
        sum_counts = {}
        prod_counts = {}
        for u, v in edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        def get_score():
            ns = len(sum_counts)
            np_ = len(prod_counts)
            return max(ns, np_)
        
        current_score = get_score()
        edges_set = set(tuple(e) for e in edges)
        
        # Precompute edge sums and products for all possible edges
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(500):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges) - 1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_u, old_v = edges[i]
                new_u, new_v = new_edge
                old_s = values[old_u] + values[old_v]
                old_p = values[old_u] * values[old_v]
                new_s = values[new_u] + values[new_v]
                new_p = values[new_u] * values[new_v]
                
                # Remove old edge
                sum_counts[old_s] -= 1
                if sum_counts[old_s] == 0:
                    del sum_counts[old_s]
                prod_counts[old_p] -= 1
                if prod_counts[old_p] == 0:
                    del prod_counts[old_p]
                
                # Add new edge
                sum_counts[new_s] = sum_counts.get(new_s, 0) + 1
                prod_counts[new_p] = prod_counts.get(new_p, 0) + 1
                
                new_score = get_score()
                if new_score < current_score:
                    current_score = new_score
                    edges[i] = [new_u, new_v]
                    edges_set.discard(tuple(old_edge := (old_u, old_v)))
                    edges_set.add(tuple(new_edge))
                    improved = True
                else:
                    # Revert
                    sum_counts[new_s] -= 1
                    if sum_counts[new_s] == 0:
                        del sum_counts[new_s]
                    prod_counts[new_p] -= 1
                    if prod_counts[new_p] == 0:
                        del prod_counts[new_p]
                    sum_counts[old_s] = sum_counts.get(old_s, 0) + 1
                    prod_counts[old_p] = prod_counts.get(old_p, 0) + 1
        
        return edges, current_score
    
    def fast_value_local_search(values, edges, rng, time_limit):
        """Fast value local search with limited candidate set."""
        sum_counts = {}
        prod_counts = {}
        for u, v in edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        def get_score():
            return max(len(sum_counts), len(prod_counts))
        
        current_score = get_score()
        value_set = set(values)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(30):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n - 1)
                old_val = values[idx]
                
                # Generate focused candidate set
                candidates = set()
                for delta in range(-5, 6):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        candidates.add(v)
                # Common multiples
                for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24]:
                    for mult in range(1, min(30, max_val // base + 1)):
                        candidates.add(base * mult)
                # Powers
                for base in [2, 3, 4]:
                    v = 1
                    while v <= max_val:
                        candidates.add(v)
                        v *= base
                
                best_new_val = None
                best_new_score = current_score
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val or new_val in value_set:
                        continue
                    
                    # Remove old contributions
                    for u, v in edges:
                        if u == idx:
                            other = v
                            old_s = old_val + values[other]
                            old_p = old_val * values[other]
                            sum_counts[old_s] -= 1
                            if sum_counts[old_s] == 0:
                                del sum_counts[old_s]
                            prod_counts[old_p] -= 1
                            if prod_counts[old_p] == 0:
                                del prod_counts[old_p]
                        elif v == idx:
                            other = u
                            old_s = values[other] + old_val
                            old_p = values[other] * old_val
                            sum_counts[old_s] -= 1
                            if sum_counts[old_s] == 0:
                                del sum_counts[old_s]
                            prod_counts[old_p] -= 1
                            if prod_counts[old_p] == 0:
                                del prod_counts[old_p]
                    
                    # Add new contributions
                    for u, v in edges:
                        if u == idx:
                            other = v
                            new_s = new_val + values[other]
                            new_p = new_val * values[other]
                            sum_counts[new_s] = sum_counts.get(new_s, 0) + 1
                            prod_counts[new_p] = prod_counts.get(new_p, 0) + 1
                        elif v == idx:
                            other = u
                            new_s = values[other] + new_val
                            new_p = values[other] * new_val
                            sum_counts[new_s] = sum_counts.get(new_s, 0) + 1
                            prod_counts[new_p] = prod_counts.get(new_p, 0) + 1
                    
                    new_score = get_score()
                    
                    # Revert
                    for u, v in edges:
                        if u == idx:
                            other = v
                            new_s = new_val + values[other]
                            new_p = new_val * values[other]
                            sum_counts[new_s] -= 1
                            if sum_counts[new_s] == 0:
                                del sum_counts[new_s]
                            prod_counts[new_p] -= 1
                            if prod_counts[new_p] == 0:
                                del prod_counts[new_p]
                        elif v == idx:
                            other = u
                            new_s = values[other] + new_val
                            new_p = values[other] * new_val
                            sum_counts[new_s] -= 1
                            if sum_counts[new_s] == 0:
                                del sum_counts[new_s]
                            prod_counts[new_p] -= 1
                            if prod_counts[new_p] == 0:
                                del prod_counts[new_p]
                    
                    for u, v in edges:
                        if u == idx:
                            other = v
                            old_s = old_val + values[other]
                            old_p = old_val * values[other]
                            sum_counts[old_s] = sum_counts.get(old_s, 0) + 1
                            prod_counts[old_p] = prod_counts.get(old_p, 0) + 1
                        elif v == idx:
                            other = u
                            old_s = values[other] + old_val
                            old_p = values[other] * old_val
                            sum_counts[old_s] = sum_counts.get(old_s, 0) + 1
                            prod_counts[old_p] = prod_counts.get(old_p, 0) + 1
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                
                if best_new_val is not None:
                    value_set.discard(old_val)
                    value_set.add(best_new_val)
                    values[idx] = best_new_val
                    # Update counts
                    for u, v in edges:
                        if u == idx:
                            other = v
                            old_s = old_val + values[other]
                            old_p = old_val * values[other]
                            sum_counts[old_s] -= 1
                            if sum_counts[old_s] == 0:
                                del sum_counts[old_s]
                            prod_counts[old_p] -= 1
                            if prod_counts[old_p] == 0:
                                del prod_counts[old_p]
                            new_s = best_new_val + values[other]
                            new_p = best_new_val * values[other]
                            sum_counts[new_s] = sum_counts.get(new_s, 0) + 1
                            prod_counts[new_p] = prod_counts.get(new_p, 0) + 1
                        elif v == idx:
                            other = u
                            old_s = values[other] + old_val
                            old_p = values[other] * old_val
                            sum_counts[old_s] -= 1
                            if sum_counts[old_s] == 0:
                                del sum_counts[old_s]
                            prod_counts[old_p] -= 1
                            if prod_counts[old_p] == 0:
                                del prod_counts[old_p]
                            new_s = values[other] + best_new_val
                            new_p = values[other] * best_new_val
                            sum_counts[new_s] = sum_counts.get(new_s, 0) + 1
                            prod_counts[new_p] = prod_counts.get(new_p, 0) + 1
                    current_score = best_new_score
                    improved = True
        
        return values, current_score
    
    def greedy_collision_edges(values, rng, time_limit):
        """Greedy edge selection maximizing sum/product collision reuse."""
        num_edges = len(all_edges)
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # Group by sum and product
        sum_groups = {}
        prod_groups = {}
        for i in range(num_edges):
            s = edge_sums[i]
            p = edge_prods[i]
            if s not in sum_groups:
                sum_groups[s] = []
            sum_groups[s].append(i)
            if p not in prod_groups:
                prod_groups[p] = []
            prod_groups[p].append(i)
        
        selected = []
        selected_set = set()
        sums_used = set()
        prods_used = set()
        
        for _ in range(m):
            if time.time() > time_limit:
                break
            best_idx = None
            best_score = float('inf')
            
            for idx in range(num_edges):
                if idx in selected_set:
                    continue
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                # Score: how many existing sums/prods this edge reuses
                score = 0
                if s in sums_used:
                    score += 100
                if p in prods_used:
                    score += 100
                # Bonus for being in a large group (more potential collisions)
                score += len(sum_groups.get(s, [])) * 5
                score += len(prod_groups.get(p, [])) * 5
                # Penalty for introducing new distinct values
                if s not in sums_used:
                    score -= 50
                if p not in prods_used:
                    score -= 50
                # Tie-break: prefer smaller values
                score -= (s + p) / 10000.0
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is None:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            sums_used.add(edge_sums[best_idx])
            prods_used.add(edge_prods[best_idx])
        
        # Fill if needed
        if len(selected) < m:
            for i in range(num_edges):
                if len(selected) >= m:
                    break
                if i not in selected_set:
                    selected.append(i)
                    selected_set.add(i)
        
        return [all_edges[i] for i in selected]
    
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(12345)
    
    # Value strategies: focus on numbers with many common factors
    value_strategies = [
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168],
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30],
        [2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    ]
    
    # Phase 1: Try each value strategy with collision-aware greedy
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 15.0:
            break
        
        edges = greedy_collision_edges(init_values, rng, deadline - 15.0)
        if len(edges) < m:
            used = set(tuple(e) for e in edges)
            for e in all_edges:
                if len(edges) >= m:
                    break
                if tuple(e) not in used:
                    edges.append(list(e))
                    used.add(tuple(e))
        
        _, _, score = evaluate(init_values, edges)
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': [list(e) for e in edges]}
        
        # Fast edge local search
        if (time.time() - start_time) < deadline - 10.0:
            edges, score_e = fast_edge_local_search(init_values, edges, rng, deadline - 10.0)
            if score_e < best_overall_score:
                best_overall_score = score_e
                best_result = {'values': list(init_values), 'edges': [list(e) for e in edges]}
        
        # Fast value local search
        if (time.time() - start_time) < deadline - 8.0:
            values = list(init_values)
            values, score_v = fast_value_local_search(values, edges, rng, deadline - 8.0)
            if score_v < best_overall_score:
                best_overall_score = score_v
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
        
        # Refine edges with new values
        if (time.time() - start_time) < deadline - 5.0:
            edges3, score3 = fast_edge_local_search(values, edges, rng, deadline - 5.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values), 'edges': [list(e) for e in edges3]}
    
    # Phase 2: Aggressive local search on best result
    if best_result is not None:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        for iteration in range(30):
            if (time.time() - start_time) > deadline - 3.0:
                break
            
            values, s1 = fast_value_local_search(values, edges, rng, deadline - 3.0)
            edges, s2 = fast_edge_local_search(values, edges, rng, deadline - 3.0)
            
            if s2 < best_overall_score:
                best_overall_score = s2
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
