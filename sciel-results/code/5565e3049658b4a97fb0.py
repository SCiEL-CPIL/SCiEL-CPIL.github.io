import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120

    # Build Clebsch graph as starting point
    # Vertices 0..15, edge if Hamming distance (in 4-bit representation) is 1 or 4
    def hamming_weight(x):
        return bin(x).count('1')

    def build_clebsch():
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                d = hamming_weight(u ^ v)
                if d == 1 or d == 4:
                    edges.append([u, v])
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
        return True

    def is_3_colorable(adj):
        # Backtracking 3-coloring
        colors = [-1] * n
        
        def backtrack(vertex):
            if vertex == n:
                return True
            for c in range(3):
                conflict = False
                for nb in adj[vertex]:
                    if colors[nb] == c:
                        conflict = True
                        break
                if not conflict:
                    colors[vertex] = c
                    if backtrack(vertex + 1):
                        return True
                    colors[vertex] = -1
            return False
        
        # Try to find a 3-coloring; if found, graph IS 3-colorable
        return backtrack(0)

    def count_edges(adj):
        return sum(len(adj[u]) for u in range(n)) // 2

    def add_edge(adj, u, v):
        """Add edge if triangle-free. Returns True if added."""
        if u in adj[v]:
            return False
        # Check no common neighbor
        for w in adj[u]:
            if w in adj[v]:
                return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    def remove_edge(adj, u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Initial: Clebsch graph
    edges, adj = build_clebsch()
    best_edges = edges
    best_adj = [set(s) for s in adj]
    best_count = len(edges)
    
    # Verify initial
    assert is_triangle_free(adj)
    initial_3c = is_3_colorable(adj)
    if not initial_3c:
        best_count = len(edges)
    
    # Simulated annealing / local search
    # We want to maximize edges while maintaining triangle-free and not-3-colorable
    # Strategy: try adding edges (preserving triangle-free), if it makes 3-colorable,
    # we can still accept with some probability (annealing), but we need to eventually
    # get back to non-3-colorable.
    
    # Actually, a better approach: 
    # Phase 1: Maximize edges while keeping triangle-free (may become 3-colorable)
    # Phase 2: Among maximal triangle-free graphs, find one that is not 3-colorable
    # with maximum edges.
    
    # Let's do a more targeted search:
    # Start from Clebsch, try all possible edge additions that preserve triangle-free.
    # For each, check if still not 3-colorable. If yes, accept.
    # If no, we might accept temporarily but need to recover.
    
    # Better: greedy + random restarts with simulated annealing on the edge set
    # Objective: maximize edges subject to triangle-free AND not 3-colorable
    
    # Let's use a different approach:
    # 1. Start with Clebsch
    # 2. Try to add edges one by one (in random order), keeping triangle-free
    # 3. After each addition, check if still not 3-colorable
    # 4. If it becomes 3-colorable, try removing a different edge to restore non-3-colorability
    # 5. Use simulated annealing to escape local optima
    
    # Simulated annealing approach:
    # State: adjacency (triangle-free)
    # Move: add an edge, remove an edge, or swap (add one, remove another)
    # Energy: -num_edges (we want to minimize energy = maximize edges)
    # Constraint: must be triangle-free. Non-3-colorable is the target property.
    
    # Let's do iterative improvement with random restarts:
    
    def try_improve(adj, max_time):
        """Try to add edges while preserving triangle-free and not-3-colorable."""
        adj = [set(s) for s in adj]
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            # Get all non-edges
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        non_edges.append((u, v))
            random.shuffle(non_edges)
            for (u, v) in non_edges:
                if (time.time() - start_time) >= max_time:
                    break
                # Check if adding preserves triangle-free
                can_add = True
                for w in adj[u]:
                    if w in adj[v]:
                        can_add = False
                        break
                if not can_add:
                    continue
                # Try adding
                adj[u].add(v)
                adj[v].add(u)
                # Check if still not 3-colorable
                if not is_3_colorable(adj):
                    improved = True
                    # Keep the edge
                else:
                    # Became 3-colorable, revert
                    adj[u].discard(v)
                    adj[v].discard(u)
        return adj

    def random_maximal_triangle_free():
        """Generate a random maximal triangle-free graph."""
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    # Try to add
                    can_add = True
                    for w in adj[u]:
                        if w in adj[v]:
                            can_add = False
                            break
                    if can_add:
                        adj[u].add(v)
                        adj[v].add(u)
        # Make it maximal: add all possible edges
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    can_add = True
                    for w in adj[u]:
                        if w in adj[v]:
                            can_add = False
                            break
                    if can_add:
                        adj[u].add(v)
                        adj[v].add(u)
        return adj

    # Main search loop
    best_adj = [set(s) for s in adj]
    best_count = count_edges(best_adj)
    
    # Try improving from Clebsch
    improved_adj = try_improve(adj, deadline - 5)
    improved_count = count_edges(improved_adj)
    if not is_3_colorable(improved_adj):
        if improved_count > best_count:
            best_adj = [set(s) for s in improved_adj]
            best_count = improved_count
    
    # Random restarts with simulated annealing
    attempts = 0
    while (time.time() - start_time) < deadline - 10:
        attempts += 1
        
        # Random restart
        if attempts % 5 == 0:
            adj = random_maximal_triangle_free()
        else:
            # Perturb current best
            adj = [set(s) for s in best_adj]
            # Remove a random edge
            all_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        all_edges.append((u, v))
            if all_edges:
                u, v = random.choice(all_edges)
                remove_edge(adj, u, v)
        
        # Try to improve
        improved_adj = try_improve(adj, deadline - 10)
        improved_count = count_edges(improved_adj)
        
        if not is_3_colorable(improved_adj):
            if improved_count > best_count:
                best_adj = [set(s) for s in improved_adj]
                best_count = improved_count
    
    # Final: try one more round of improvement from best
    if (time.time() - start_time) < deadline - 5:
        final_adj = try_improve(best_adj, deadline - 2)
        final_count = count_edges(final_adj)
        if not is_3_colorable(final_adj) and final_count > best_count:
            best_adj = [set(s) for s in final_adj]
            best_count = final_count
    
    return get_edges_from_adj(best_adj)
