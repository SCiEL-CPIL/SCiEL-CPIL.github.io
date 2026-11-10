"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    
    deadline = time.time() + 85.0
    N = 256
    universe = list(range(2, N + 1))
    n = len(universe)
    
    # Build conflict graph
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        ai = universe[i]
        for j in range(i + 1, n):
            aj = universe[j]
            if math.lcm(ai, aj) <= N:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Weights: 1/x
    weights = [1.0 / universe[i] for i in range(n)]
    
    # Precompute neighbor sets
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and conflict[i][j]:
                neighbors[i].add(j)
    
    def reciprocal_sum(indices):
        return sum(weights[i] for i in indices)
    
    # Greedy: sort by weight descending, add if no conflict
    order_by_weight = sorted(range(n), key=lambda i: -weights[i])
    
    def greedy(order):
        selected = []
        selected_set = set()
        for i in order:
            if all(not conflict[i][j] for j in selected):
                selected.append(i)
                selected_set.add(i)
        return selected
    
    # Get initial solution from greedy
    best_indices = greedy(order_by_weight)
    best_score = reciprocal_sum(best_indices)
    
    # Local search: try adds and swaps
    def local_search(indices, score):
        nonlocal deadline
        improved = True
        cur_set = set(indices)
        cur_score = score
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding a node
            for i in range(n):
                if i in cur_set:
                    continue
                if all(not conflict[i][j] for j in cur_set):
                    new_score = cur_score + weights[i]
                    if new_score > cur_score:
                        cur_set.add(i)
                        cur_score = new_score
                        improved = True
                        break
            
            if not improved:
                # Try swaps: remove one, add another
                for i in list(cur_set):
                    if time.time() > deadline:
                        break
                    cur_set.remove(i)
                    base = cur_score - weights[i]
                    
                    best_add = None
                    best_add_score = base
                    for j in range(n):
                        if j in cur_set:
                            continue
                        if all(not conflict[j][k] for k in cur_set):
                            s = base + weights[j]
                            if s > best_add_score:
                                best_add_score = s
                                best_add = j
                        if time.time() > deadline:
                            break
                    
                    if best_add is not None and best_add_score > cur_score:
                        cur_set.add(best_add)
                        cur_score = best_add_score
                        improved = True
                    else:
                        cur_set.add(i)
                    
                    if improved:
                        break
        
        return list(cur_set), cur_score
    
    best_indices, best_score = local_search(best_indices, best_score)
    
    # Random restarts
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline:
            break
        order = list(range(n))
        random.shuffle(order)
        sol = greedy(order)
        s = reciprocal_sum(sol)
        if s > best_score:
            best_indices, best_score = local_search(sol, s)
    
    # Try branch and bound if we have time
    if time.time() < deadline - 15:
        def branch_and_bound(included, excluded, available, current_score):
            nonlocal best_indices, best_score
            
            if time.time() > deadline:
                return
            
            if current_score > best_score:
                best_score = current_score
                best_indices = list(included)
            
            if not available:
                return
            
            # Upper bound
            ub = current_score + sum(weights[i] for i in available)
            if ub <= best_score + 1e-15:
                return
            
            # Choose node with highest weight
            node = max(available, key=lambda i: weights[i])
            
            # Branch include
            new_avail_1 = set(available)
            new_avail_1.discard(node)
            for nb in neighbors[node]:
                new_avail_1.discard(nb)
            
            branch_and_bound(included | {node}, excluded, new_avail_1, current_score + weights[node])
            
            # Branch exclude
            new_avail_2 = set(available)
            new_avail_2.discard(node)
            
            branch_and_bound(included, excluded | {node}, new_avail_2, current_score)
        
        all_avail = set(range(n))
        branch_and_bound(set(), set(), all_avail, 0.0)
    
    return [universe[i] for i in best_indices]
# EVOLVE-BLOCK-END
