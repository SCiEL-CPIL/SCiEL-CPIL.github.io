"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    def evaluate_edge_set(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    def greedy_edge_selection(vals):
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        current_sums = set()
        current_products = set()
        selected_edges = []
        remaining_edges = set(all_edges)
        
        for _ in range(m):
            best_edge = None
            best_score_edge = float('inf')
            best_new_sums = set()
            best_new_products = set()
            
            for edge in remaining_edges:
                i, j = edge
                s = vals[i] + vals[j]
                p = vals[i] * vals[j]
                
                new_sums = current_sums | {s}
                new_products = current_products | {p}
                
                score = max(len(new_sums), len(new_products))
                
                if score < best_score_edge:
                    best_score_edge = score
                    best_edge = edge
                    best_new_sums = new_sums
                    best_new_products = new_products
                elif score == best_score_edge:
                    # Tie-breaker: prefer edges that add fewer new items
                    add_s = len(new_sums - current_sums)
                    add_p = len(new_products - current_products)
                    old_add_s = len(best_new_sums - current_sums)
                    old_add_p = len(best_new_products - current_products)
                    
                    if (add_s, add_p) < (old_add_s, old_add_p):
                        best_edge = edge
                        best_new_sums = new_sums
                        best_new_products = new_products
                        
            if best_edge is None:
                break
                
            selected_edges.append(list(best_edge))
            current_sums = best_new_sums
            current_products = best_new_products
            remaining_edges.remove(best_edge)
            
        return selected_edges, current_sums, current_products

    def local_search_edges(vals, initial_edges):
        # Local search to improve the edge set
        edges = [list(e) for e in initial_edges]
        edge_set = set(tuple(e) for e in edges)
        current_score = evaluate_edge_set(vals, edges)
        current_sums = set()
        current_prods = set()
        for i, j in edges:
            current_sums.add(vals[i] + vals[j])
            current_prods.add(vals[i] * vals[j])
            
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        available_edges = [e for e in all_edges if tuple(e) not in edge_set]
        
        improved = True
        while improved and time.time() - start_time < time_limit * 0.8:
            improved = False
            # Try swapping one edge in 'edges' with one in 'available_edges'
            random.shuffle(edges)
            random.shuffle(available_edges)
            
            for i in range(len(edges)):
                e_out = edges[i]
                # Calculate sums/prods without e_out
                sums_wo = set()
                prods_wo = set()
                for k, l in edges:
                    if k == e_out[0] and l == e_out[1]:
                        continue
                    sums_wo.add(vals[k] + vals[l])
                    prods_wo.add(vals[k] * vals[l])
                
                best_e_in = None
                best_new_score = current_score
                best_new_sums = None
                best_new_prods = None
                
                for e_in in available_edges:
                    if e_in == e_out:
                        continue
                    s_new = vals[e_in[0]] + vals[e_in[1]]
                    p_new = vals[e_in[0]] * vals[e_in[1]]
                    
                    new_sums = sums_wo | {s_new}
                    new_prods = prods_wo | {p_new}
                    new_score = max(len(new_sums), len(new_prods))
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_e_in = e_in
                        best_new_sums = new_sums
                        best_new_prods = new_prods
                        
                if best_e_in is not None:
                    # Perform swap
                    edges[i] = list(best_e_in)
                    edge_set.remove(tuple(e_out))
                    edge_set.add(tuple(best_e_in))
                    
                    # Update available edges
                    available_edges.remove(best_e_in)
                    available_edges.append(e_out)
                    
                    current_score = best_new_score
                    current_sums = best_new_sums
                    current_prods = best_new_prods
                    improved = True
                    if time.time() - start_time > time_limit * 0.8:
                        break
                        
        return edges, current_score

    # Generate candidate value sets
    candidates = []
    
    # 1. Small integers (High collision probability)
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, n + 2)))
    candidates.append(list(range(3, n + 3)))
    
    # 2. Arithmetic Progressions
    for start in [1, 2, 5, 10]:
        for step in [1, 2, 3, 5]:
            vals = [start + i * step for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidates.append(vals)
                
    # 3. Multiples of small numbers
    for base in [2, 3, 4, 5]:
        vals = [base * (i + 1) for i in range(n)]
        if max(vals) <= 10000 and len(set(vals)) == n:
            candidates.append(vals)
            
    # 4. Random small integers
    for _ in range(20):
        vals = random.sample(range(1, 50), n)
        candidates.append(vals)
        
    # 5. Random mixed
    for _ in range(20):
        vals = random.sample(range(1, 100), n)
        candidates.append(vals)

    best_vals = None
    best_edges = None
    best_score = float('inf')
    
    # Evaluate candidates
    for vals in candidates:
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > 10000:
            continue
            
        # Quick check: if the total distinct sums/products in the full graph is already > best_score, skip?
        # No, we only pick 35 edges, so we can avoid some.
        
        # Greedy selection
        edges, sums, prods = greedy_edge_selection(vals)
        score = max(len(sums), len(prods))
        
        if score < best_score:
            best_score = score
            best_vals = list(vals)
            best_edges = [list(e) for e in edges]
            
        # Local search on edges
        if time.time() - start_time < time_limit * 0.7:
            improved_edges, improved_score = local_search_edges(vals, edges)
            if improved_score < best_score:
                best_score = improved_score
                best_vals = list(vals)
                best_edges = improved_edges

    if best_vals is None:
        best_vals = list(range(1, n + 1))
        best_edges, _, _ = greedy_edge_selection(best_vals)

    return {'values': best_vals, 'edges': best_edges}
# EVOLVE-BLOCK-END
