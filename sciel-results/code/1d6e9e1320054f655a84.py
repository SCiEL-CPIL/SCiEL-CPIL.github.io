def construct():
    import time
    import random

    n = 16
    time_limit = 110.0
    start_time = time.time()

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def make_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def is_triangle_free_edges(edge_set):
        adj = make_adj(edge_set)
        return is_triangle_free(adj)

    def is_3_colorable(edge_set):
        """Check if the graph is 3-colorable using backtracking with MRV heuristic."""
        adj = make_adj(edge_set)
        color = [-1] * n

        # Use MRV (Minimum Remaining Values) heuristic for ordering
        # Pre-compute degree for initial ordering
        degrees = [len(adj[v]) for v in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])

        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try colors in order of least constraining
            for c in range(3):
                ok = True
                for nei in adj[v]:
                    if color[nei] == c:
                        ok = False
                        break
                if ok:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False

        return backtrack(0)

    def try_add_edges_greedy(edge_set):
        """Try to add as many edges as possible while maintaining triangle-freedom."""
        edge_set = set(edge_set)
        adj = make_adj(edge_set)
        
        # Keep trying until no more edges can be added
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if u not in adj[v]:
                    # Check if adding (u,v) creates a triangle
                    if not (adj[u] & adj[v]):
                        edge_set.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        improved = True
        return edge_set

    # Start with a known non-3-colorable triangle-free graph on 11 vertices (Grötzsch)
    # and extend it.
    # Grötzsch graph: 11 vertices
    # Vertices 0-4 form a 5-cycle
    # Vertices 5-9 are "shadow" vertices
    # Vertex 10 is connected to all of 5-9
    
    groetzsch_edges = set()
    # 5-cycle: 0-1-2-3-4-0
    for i in range(5):
        groetzsch_edges.add((i, (i + 1) % 5))
    # Mycielski construction:
    # For each edge (u, v) in the cycle, vertex 5+u connects to v and vertex 5+v connects to u
    # Actually, standard Mycielski:
    # New vertices 5..9 (let's call them 5+i for i in 0..4)
    # 5+i is connected to i and (i+1)%5? No.
    # In Mycielski(G), for each vertex v in G, we create v'. v' is connected to all neighbors of v.
    # Here G is C5.
    # So vertex 5+i (corresponding to vertex i) is connected to neighbors of i in C5.
    # Neighbors of i are (i-1)%5 and (i+1)%5.
    for i in range(5):
        groetzsch_edges.add((5 + i, (i - 1) % 5))
        groetzsch_edges.add((5 + i, (i + 1) % 5))
    # Vertex 10 is connected to all 5+i
    for i in range(5):
        groetzsch_edges.add((10, 5 + i))

    best_edge_set = set()
    best_count = 0

    # Verify Grötzsch
    if is_triangle_free_edges(groetzsch_edges) and not is_3_colorable(groetzsch_edges):
        best_edge_set = groetzsch_edges
        best_count = len(groetzsch_edges)
        # Greedily complete
        completed = try_add_edges_greedy(groetzsch_edges)
        if is_triangle_free_edges(completed) and not is_3_colorable(completed):
            if len(completed) > best_count:
                best_count = len(completed)
                best_edge_set = completed

    # Randomized local search
    random.seed(42)
    time_limit = 110.0
    
    def evaluate_and_update(edge_set):
        nonlocal best_edge_set, best_count
        if is_triangle_free_edges(edge_set) and not is_3_colorable(edge_set):
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_edge_set = set(edge_set)
                return True
        return False

    # Phase 1: Random greedy constructions
    for _ in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly shuffle edges and greedily add
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        edge_set = set()
        adj = [set() for _ in range(n)]
        for u, v in shuffled:
            if u not in adj[v]:
                if not (adj[u] & adj[v]):
                    edge_set.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Check if non-3-colorable
        if not is_3_colorable(edge_set):
            # Try to complete
            completed = try_add_edges_greedy(edge_set)
            if is_triangle_free_edges(completed) and not is_3_colorable(completed):
                evaluate_and_update(completed)
            
            # Also try local search from this point
            # Try removing an edge and re-completing
            if edge_set:
                edge_list = list(edge_set)
                for _ in range(10):
                    if time.time() - start_time > time_limit:
                        break
                    e_to_remove = random.choice(edge_list)
                    new_set = set(edge_set)
                    new_set.discard(e_to_remove)
                    new_set.discard((e_to_remove[1], e_to_remove[0]))
                    completed = try_add_edges_greedy(new_set)
                    if is_triangle_free_edges(completed) and not is_3_colorable(completed):
                        evaluate_and_update(completed)
    
    # Phase 2: Simulated Annealing / Hill Climbing on the best solution found
    if best_edge_set:
        current_set = set(best_edge_set)
        current_count = len(current_set)
        T = 1.0
        cooling = 0.999
        
        while time.time() - start_time < time_limit and T > 0.01:
            # Pick a random move: add edge, remove edge, or swap
            move_type = random.random()
            new_set = set(current_set)
            
            if move_type < 0.3 and len(new_set) < n*(n-1)//2:
                # Add a random edge
                candidates = [e for e in all_edges if e not in new_set and (e[1], e[0]) not in new_set]
                if candidates:
                    e = random.choice(candidates)
                    new_set.add(e)
            elif move_type < 0.6 and new_set:
                # Remove a random edge
                e = random.choice(list(new_set))
                new_set.discard(e)
            else:
                # Swap: remove one, try to add another
                if new_set:
                    e_rem = random.choice(list(new_set))
                    new_set.discard(e_rem)
                    candidates = [e for e in all_edges if e not in new_set and (e[1], e[0]) not in new_set]
                    if candidates:
                        e_add = random.choice(candidates)
                        new_set.add(e_add)
            
            # Check validity
            if is_triangle_free_edges(new_set):
                if not is_3_colorable(new_set):
                    # Accept if better, or with probability exp(-delta/T)
                    delta = len(new_set) - len(current_set)
                    if delta > 0 or random.random() < 0.5:  # Simplified acceptance
                        current_set = new_set
                        evaluate_and_update(current_set)
                else:
                    # It's 3-colorable, reject unless we're in a "descent" phase (but we want non-colorable)
                    # So reject
                    pass
            T *= cooling
            
            # Periodic restart from best
            if random.random() < 0.01:
                current_set = set(best_edge_set)
                T = 1.0

    return [list(e) for e in best_edge_set]
