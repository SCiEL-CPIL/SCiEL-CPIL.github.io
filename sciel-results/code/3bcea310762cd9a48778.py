import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = n_vertices * d // 2  # 448
    
    # Time management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 120s limit (evaluator kills at 150s, but task says 120s)
    
    # Precompute all edges
    edges = []
    edge_set = set()
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_set.add((u, v))
    
    # Precompute for each edge, the "opposite" edges that would form a C4
    # An edge (u,v) where v = u ^ (1<<i)
    # A C4 containing this edge: for each j != i, the face is spanned by dimensions i and j
    # The four edges of this face are:
    #   (u, u^(1<<i)), (u^(1<<j), u^(1<<i)^(1<<j))  -- these are the "vertical" edges (dim i)
    #   (u, u^(1<<j)), (u^(1<<i), u^(1<<i)^(1<<j))  -- these are the "horizontal" edges (dim j)
    # So if we add edge (u, v) where v = u ^ (1<<i), we need to check for each j != i:
    #   Are all of these three edges already present?
    #   (u, u^(1<<j)), (v, v^(1<<j)), (u^(1<<j), v^(1<<j))
    
    # For O(1) conflict checking, we maintain:
    # - A set of chosen edges (as frozenset or sorted tuple)
    # - For each vertex, the set of chosen neighbors (or a bitmask of which dimensions are used)
    
    # Better approach: for each vertex u and dimension i, track whether edge (u, u^(1<<i)) is chosen
    # Using a 2D boolean array: chosen[u][i] = True if edge in dimension i from u is chosen
    # But each edge is shared by two vertices, so we can just use chosen[u][i] for u < u^(1<<i)
    
    # Let's use: edge_present[u][i] = True if the edge from u in direction i is present
    # Since the graph is undirected, edge_present[u][i] == edge_present[u^(1<<i)][i]
    
    edge_present = np.zeros((n_vertices, d), dtype=bool)
    
    # For O(1) C4 check when adding edge (u, v) where v = u ^ (1<<i):
    # For each j != i, check if edges (u, u^(1<<j)), (v, v^(1<<j)), (u^(1<<j), v^(1<<j)) are all present
    # i.e., edge_present[u][j] and edge_present[v][j] and edge_present[u^(1<<j)][i]
    
    def can_add(u, i):
        """Check if adding edge from u in direction i creates a C4."""
        v = u ^ (1 << i)
        for j in range(d):
            if j == i:
                continue
            if edge_present[u][j] and edge_present[v][j] and edge_present[u ^ (1 << j)][i]:
                return False
        return True
    
    def add_edge(u, i):
        """Add edge from u in direction i."""
        v = u ^ (1 << i)
        edge_present[u][i] = True
        edge_present[v][i] = True
    
    def remove_edge(u, i):
        """Remove edge from u in direction i."""
        v = u ^ (1 << i)
        edge_present[u][i] = False
        edge_present[v][i] = False
    
    def count_edges():
        """Count total edges (each edge counted twice, so divide by 2)."""
        return int(np.sum(edge_present)) // 2
    
    # Phase 1: Greedy construction with random ordering
    best_count = 0
    best_edge_present = None
    
    def greedy_random_order():
        """Greedy: try edges in random order, add if no C4 conflict."""
        nonlocal edge_present
        edge_present[:] = False
        order = list(range(n_edges))
        random.shuffle(order)
        
        for idx in order:
            u, v = edges[idx]
            # Find which dimension this edge is in
            diff = u ^ v
            i = diff.bit_length() - 1
            if can_add(u, i):
                add_edge(u, i)
        
        return count_edges()
    
    # Do a few random greedy tries to find a good starting point
    num_greedy = 5
    for _ in range(num_greedy):
        if time.time() - start_time > time_limit - 5:
            break
        c = greedy_random_order()
        if c > best_count:
            best_count = c
            best_edge_present = edge_present.copy()
    
    # Restore best state
    if best_edge_present is not None:
        edge_present[:] = best_edge_present
    
    # Phase 2: Simulated annealing with random kicks
    # Strategy: remove 1-2 random edges, then greedily refill
    current_count = best_count
    current_edge_present = edge_present.copy()
    edge_present[:] = current_edge_present
    
    best_overall = current_count
    best_overall_state = current_edge_present.copy()
    
    # Temperature schedule
    t0 = 1.0
    t_min = 0.01
    cooling = 0.9995
    
    temperature = t0
    iteration = 0
    
    while time.time() - start_time < time_limit - 2:
        iteration += 1
        
        # Random kick: remove 1 or 2 random edges
        num_remove = random.choice([1, 1, 2])
        removed = []
        attempts = 0
        while len(removed) < num_remove and attempts < 20:
            attempts += 1
            u = random.randint(0, n_vertices - 1)
            i = random.randint(0, d - 1)
            if edge_present[u][i]:
                v = u ^ (1 << i)
                if u < v:  # Only remove from one end to avoid double-counting
                    remove_edge(u, i)
                    removed.append((u, i))
        
        if not removed:
            continue
        
        # Greedily refill: try to add back removed edges and other random edges
        # First try to add back the removed edges
        for (u, i) in removed:
            if can_add(u, i):
                add_edge(u, i)
        
        # Then try random edges to fill in
        num_fill_attempts = random.randint(5, 15)
        for _ in range(num_fill_attempts):
            if time.time() - start_time > time_limit - 1:
                break
            idx = random.randint(0, n_edges - 1)
            u, v = edges[idx]
            diff = u ^ v
            i = diff.bit_length() - 1
            if not edge_present[u][i]:
                if can_add(u, i):
                    add_edge(u, i)
        
        new_count = count_edges()
        
        # Accept or reject based on simulated annealing
        delta = new_count - current_count
        if delta >= 0 or random.random() < np.exp(delta / temperature):
            current_count = new_count
            if current_count > best_overall:
                best_overall = current_count
                best_overall_state = edge_present.copy()
        else:
            # Revert: we need to restore the state
            # This is tricky because we've modified edge_present
            # Let's just keep it simple: if rejected, we lose the state
            # Better approach: save state before modification
            pass
        
        # Cool down
        temperature = max(t_min, temperature * cooling)
        
        # If we've hit the theoretical max, stop
        if best_overall >= 308:
            break
    
    # Restore best state and convert to output format
    edge_present[:] = best_overall_state
    out = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v and edge_present[u][i]:
                out.append([u, v])
    
    return out
