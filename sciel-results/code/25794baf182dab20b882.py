import random
import itertools
import time

def construct():
    n = 14
    m = 35
    
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, n + 2)))
    candidates.append(list(range(1, n + 1, 2))) # Odd numbers
    candidates.append(list(range(2, n + 1, 2))) # Even numbers
    
    # 2. Multiples
    candidates.append([2 * i for i in range(1, n + 1)])
    candidates.append([3 * i for i in range(1, n + 1)])
    
    # 3. Structured sets with collisions
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 20, 24])
    candidates.append([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 24])
    
    # 4. Random sets
    random.seed(42)
    for _ in range(30):
        vals = random.sample(range(1, 50), n)
        candidates.append(vals)

    best_score = float('inf')
    best_edges = None
    best_values = None
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def greedy_select_edges(values):
        all_edges = list(itertools.combinations(range(n), 2))
        
        current_sums = set()
        current_prods = set()
        selected_edges = []
        remaining_edges = set(all_edges)
        
        for _ in range(m):
            best_edge = None
            best_cost = float('inf')
            
            for edge in remaining_edges:
                i, j = edge
                s = values[i] + values[j]
                p = values[i] * values[j]
                
                new_sums_count = len(current_sums) + (1 if s not in current_sums else 0)
                new_prods_count = len(current_prods) + (1 if p not in current_prods else 0)
                
                cost = max(new_sums_count, new_prods_count)
                
                if cost < best_cost:
                    best_cost = cost
                    best_edge = edge
            
            if best_edge is None:
                break
                
            i, j = best_edge
            current_sums.add(values[i] + values[j])
            current_prods.add(values[i] * values[j])
            selected_edges.append(best_edge)
            remaining_edges.remove(best_edge)
            
        return selected_edges, best_cost

    def local_search_best_improvement(values, edges, max_time=5.0):
        current_edges = set(edges)
        current_score = evaluate(values, current_edges)
        all_edges = list(itertools.combinations(range(n), 2))
        non_edges = set(all_edges) - current_edges
        start_time = time.time()
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            best_swap = None
            best_new_score = current_score
            
            # Try all possible single-edge swaps
            for e_in in current_edges:
                for e_out in non_edges:
                    new_edges = current_edges.copy()
                    new_edges.remove(e_in)
                    new_edges.add(e_out)
                    
                    new_score = evaluate(values, new_edges)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (e_in, e_out)
            
            if best_swap:
                e_in, e_out = best_swap
                current_edges.remove(e_in)
                current_edges.add(e_out)
                non_edges.remove(e_out)
                non_edges.add(e_in)
                current_score = best_new_score
                improved = True
        
        return current_score, list(current_edges)

    for values in candidates:
        if len(set(values)) != n or max(values) > 10000 or min(values) < 1:
            continue
            
        edges, score = greedy_select_edges(values)
        
        # Run local search on the greedy result
        ls_score, ls_edges = local_search_best_improvement(values, edges, max_time=2.0)
        
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
            best_values = values
            
        if best_score <= 10: # Very good
             break

    # Final polish
    if best_edges is not None:
        # Use remaining time for a longer local search
        time_left = 50 - time.time()
        if time_left > 1:
            final_score, final_edges = local_search_best_improvement(best_values, best_edges, max_time=time_left)
            if final_score < best_score:
                best_score = final_score
                best_edges = final_edges

    if best_edges is None:
        best_values = list(range(1, n + 1))
        all_edges = list(itertools.combinations(range(n), 2))
        best_edges = all_edges[:m]

    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
