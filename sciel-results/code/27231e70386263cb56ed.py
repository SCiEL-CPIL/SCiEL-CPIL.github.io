<<<<<<< SEARCH
    # Conflict graph for independent set (intersection >= 3)
    # We want to maximize the number of selected supports such that intersection <= 2.
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(N_S):
            if i != j and inter[i, j] >= 3:
                adj[i].add(j)
                adj[j].add(i)

    def greedy_mis(adj, order):
        in_set = [False] * N_S
        available = set(range(N_S))
        for v in order:
            if v in available:
                in_set[v] = True
                # Remove v and its neighbors
                to_remove = {v} | adj[v]
                available -= to_remove
        return in_set

    def local_search_mis(in_set, adj):
        # Try to add nodes
        changed = True
        while changed:
            changed = False
            for v in range(N_S):
                if not in_set[v]:
                    # Check if any neighbor is in set
                    conflict = False
                    for u in adj[v]:
                        if in_set[u]:
                            conflict = True
                            break
                    if not conflict:
                        in_set[v] = True
                        changed = True
        return in_set

    best_size = 0
    best_set = []
    rng = random.Random(42)

    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Random order
        order = list(range(N_S))
        rng.shuffle(order)

        in_set = greedy_mis(adj, order)
        in_set = local_search_mis(in_set, adj)

        size = sum(in_set)
        if size > best_size:
            best_size = size
            best_set = [i for i, b in enumerate(in_set) if b]
            # Optional: try to improve best_set further with local search
            # But local_search_mis is already applied.
=======
    # Build adjacency matrix for vectorized operations
    adj_matrix = (inter >= 3)
    np.fill_diagonal(adj_matrix, False)
    adj_matrix = adj_matrix.astype(np.int8)
    
    def greedy_mis_vectorized(order):
        in_set = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        for v in order:
            if available[v] and not in_set[v]:
                in_set[v] = True
                # Remove v and its neighbors from available
                available[v] = False
                neighbors = np.where(adj_matrix[v] == 1)[0]
                available[neighbors] = False
        return in_set

    def local_search_mis_vectorized(in_set):
        # Try to add nodes that have no neighbors in set
        changed = True
        while changed:
            changed = False
            # For each node not in set, check if it has any neighbor in set
            neighbor_count = in_set @ adj_matrix
            can_add = ~in_set & (neighbor_count == 0)
            if np.any(can_add):
                in_set[can_add] = True
                changed = True
        
        # Try swap moves: remove one node, add up to two nodes that were blocked only by it
        for _ in range(3):
            improved = False
            # For each node in set, try removing it and see how many new nodes we can add
            for v in range(N_S):
                if not in_set[v]:
                    continue
                # Temporarily remove v
                in_set[v] = False
                neighbor_count = in_set @ adj_matrix
                can_add = ~in_set & (neighbor_count == 0)
                new_count = np.sum(can_add)
                if new_count >= 2:
                    # Add all compatible nodes
                    in_set[can_add] = True
                    improved = True
                    break
                else:
                    # Revert
                    in_set[v] = True
            
            if not improved:
                break
        
        return in_set

    def branch_and_bound_mis(adj_matrix, timeout=2.0):
        """Branch and bound for maximum independent set"""
        start = time.time()
        best_set = np.zeros(N_S, dtype=bool)
        best_size = 0
        
        # Upper bound: greedy coloring on remaining graph
        def greedy_color_bound(remaining):
            """Upper bound on MIS size using greedy coloring"""
            if not np.any(remaining):
                return 0
            # Greedy coloring
            color = np.full(N_S, -1, dtype=np.int32)
            uncolored = remaining.copy()
            num_colors = 0
            while np.any(uncolored):
                # Pick first uncolored vertex
                v = np.where(uncolored)[0][0]
                # Assign color
                color[v] = num_colors
                uncolored[v] = False
                # Remove neighbors from uncolored
                neighbors = np.where(adj_matrix[v] == 1)[0]
                uncolored[neighbors] = False
                num_colors += 1
            return num_colors
        
        def branch_bound(remaining, current_set, depth=0):
            nonlocal best_set, best_size
            
            if time.time() - start > timeout:
                return
            
            if not np.any(remaining):
                if np.sum(current_set) > best_size:
                    best_size = np.sum(current_set)
                    best_set = current_set.copy()
                return
            
            # Upper bound
            upper_bound = np.sum(current_set) + greedy_color_bound(remaining)
            if upper_bound <= best_size:
                return
            
            # Pick vertex with highest degree in remaining graph
            degrees = adj_matrix[remaining] @ remaining
            v = np.where(remaining)[0][np.argmax(degrees[np.where(remaining)[0]])]
            
            # Branch 1: Include v
            new_remaining = remaining.copy()
            new_remaining[v] = False
            neighbors = np.where(adj_matrix[v] == 1)[0]
            new_remaining[neighbors] = False
            new_current = current_set.copy()
            new_current[v] = True
            branch_bound(new_remaining, new_current, depth + 1)
            
            # Branch 2: Exclude v
            new_remaining = remaining.copy()
            new_remaining[v] = False
            branch_bound(new_remaining, current_set, depth + 1)
        
        initial_remaining = np.ones(N_S, dtype=bool)
        branch_bound(initial_remaining, np.zeros(N_S, dtype=bool))
        
        return best_set

    # First, use branch and bound to get a good initial solution
    initial_set = branch_and_bound_mis(adj_matrix, timeout=10.0)
    
    best_size = np.sum(initial_set)
    best_set_idx = np.where(initial_set)[0].tolist()
    
    rng = random.Random(42)
    
    # Then use local search with multiple restarts to try to improve
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Random order
        order = list(range(N_S))
        rng.shuffle(order)

        in_set = greedy_mis_vectorized(order)
        in_set = local_search_mis_vectorized(in_set)

        size = np.sum(in_set)
        if size > best_size:
            best_size = size
            best_set_idx = np.where(in_set)[0].tolist()
>>>>>>> REPLACE

