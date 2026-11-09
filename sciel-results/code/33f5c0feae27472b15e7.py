def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill

    N = 17
    MIN_D = 6
    SIZE = 1 << N

    # Precompute weights for all 17-bit vectors
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    def try_greedy_linear(order, time_limit):
        current_code = [0]
        basis = []
        target_basis_size = 8

        for g in order:
            if time.time() > time_limit:
                break
            if len(basis) >= target_basis_size:
                break

            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break

            if valid:
                basis.append(g)
                new_words = [c ^ g for c in current_code]
                current_code.extend(new_words)

        return current_code

    # Generate candidate order for linear code: sorted by weight descending
    indices = list(range(1, SIZE))
    indices.sort(key=lambda x: -weights[x])

    # Phase 1: Find a 256-word linear code
    code = try_greedy_linear(indices, deadline - 60)
    best_code = code
    best_size = len(code)

    if best_size < 256:
        for _ in range(5):
            if time.time() > deadline - 60:
                break
            random.shuffle(indices)
            code = try_greedy_linear(indices, deadline - 60)
            if len(code) > best_size:
                best_code = code
                best_size = len(code)

    C = best_code
    C_set = set(C)
    C_np = np.array(C, dtype=np.int32)
    
    # Phase 2: Find all vectors valid for extension
    # A vector w is valid if d(w, c) >= 6 for all c in C
    valid_candidates = []
    
    t_start_phase2 = time.time()
    time_limit_phase2 = t_start_phase2 + 20
    
    # Vectorized check for all w? 
    # 131k * 256 is 33M ops. Numpy can do this in chunks.
    # But Python loop over 131k might be okay if inner check is vectorized.
    # Let's try to optimize the inner loop.
    
    for w in range(SIZE):
        if time.time() > time_limit_phase2:
            break
        if w in C_set:
            continue
        
        # Check distance to C
        dists = weights[w ^ C_np]
        if dists.min() >= MIN_D:
            valid_candidates.append(w)

    if not valid_candidates:
        return best_code

    # Phase 3: Find Maximum Independent Set in the graph of valid_candidates
    # Vertices: valid_candidates
    # Edge: d(u, v) < 6
    
    V = valid_candidates
    V_len = len(V)
    
    # Build adjacency list
    # For each u in V, find v in V such that d(u,v) < 6
    # We can use the weights table.
    
    # To speed up MIS, we can use a greedy coloring heuristic (DSATUR-like or simple degree-based).
    # Since V_len is likely small (< 1000?), we can afford O(V^2) preprocessing.
    
    adj = [set() for _ in range(V_len)]
    
    # Build graph
    for i in range(V_len):
        u = V[i]
        for j in range(i + 1, V_len):
            v = V[j]
            if weights[u ^ v] < MIN_D:
                adj[i].add(j)
                adj[j].add(i)

    # Solve MIS using greedy coloring / pruning
    # Strategy: Iteratively pick a vertex, add to IS, remove it and its neighbors from graph.
    # To get a good IS, we should pick vertices with low degree (more likely to be compatible with others).
    
    best_is = []
    
    # Multiple random restarts of the greedy MIS
    # We can also try different orderings.
    
    time_limit_phase3 = deadline - 2
    
    # Heuristic: Sort vertices by degree ascending initially?
    # Or just random shuffle and greedy pick.
    
    # Let's try a few strategies:
    # 1. Greedy by degree (ascending)
    # 2. Random Greedy
    
    def greedy_mis(adj_copy, order_func):
        """
        adj_copy: list of sets (mutable copy or we work on a list of alive vertices)
        order_func: function to pick next vertex
        """
        # We maintain a list of alive vertices
        alive = set(range(V_len))
        is_set = set()
        
        # Copy adjacency for alive vertices? 
        # It's easier to just check if neighbor is in alive.
        
        while alive:
            # Pick a vertex
            # If order_func is 'degree', pick min degree
            # If 'random', pick random
            
            if order_func == 'degree':
                # Find vertex with min degree in alive subgraph
                min_deg = float('inf')
                best_v = -1
                for v in alive:
                    deg = len(adj[v] & alive)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                v = best_v
            elif order_func == 'random':
                v = random.choice(list(alive))
            else:
                v = next(iter(alive))
                
            if v == -1:
                break
                
            is_set.add(v)
            # Remove v and its neighbors
            to_remove = {v} | (adj[v] & alive)
            alive -= to_remove
            
        return [V[i] for i in is_set]

    # Try degree-based greedy
    # Note: adj is static, but we intersect with alive.
    
    # To make it faster, we can pre-compute degrees or just use the set intersection.
    # V_len is small enough.
    
    best_extension = []
    
    # Run multiple restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > time_limit_phase3:
            break
            
        # Alternate between degree and random
        if r % 2 == 0:
            ext = greedy_mis(adj, 'degree')
        else:
            ext = greedy_mis(adj, 'random')
            
        if len(ext) > len(best_extension):
            best_extension = ext
            
    # Combine
    final_code = best_code + best_extension
    
    return final_code
