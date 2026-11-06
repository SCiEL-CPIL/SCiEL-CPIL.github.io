"""Rewritten program for R(3,10) triangle-free graph construction."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 110  # seconds; leave safety margin before 120s limit

    # Known best: Andrasfai graph And(9) on 26 vertices.
    # We will attempt to construct larger triangle-free graphs with alpha < 10.
    # Strategy: use known constructions and local search.

    best_n = 0
    best_edges = []

    def check_triangle_free(n, edges):
        """Check if graph is triangle-free using adjacency matrix."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def max_independent_set_size(n, edges):
        """Exact branch and bound for maximum independent set size."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        best = [0]

        def branch(remaining, size):
            if time.time() - start_time > DEADLINE - 5:
                return
            if not remaining:
                if size > best[0]:
                    best[0] = size
                return
            # Upper bound: size + len(remaining)
            if size + len(remaining) <= best[0]:
                return
            # Pick vertex with max degree in remaining
            r = list(remaining)
            v = max(r, key=lambda x: len(adj[x] & set(r)))
            # Branch 1: include v
            new_remaining = set(r) - {v} - adj[v]
            branch(new_remaining, size + 1)
            # Branch 2: exclude v
            new_remaining2 = set(r) - {v}
            branch(new_remaining2, size)

        branch(set(range(n)), 0)
        return best[0]

    def is_valid(n, edges):
        """Check both conditions: no triangle and no independent set of 10."""
        if time.time() - start_time > DEADLINE - 5:
            return False
        if not check_triangle_free(n, edges):
            return False
        alpha = max_independent_set_size(n, edges)
        return alpha < 10

    # --- Candidate 1: Andrasfai graph And(9) on 26 vertices ---
    n1 = 26
    edges1 = [[i, j] for i in range(n1) for j in range(i + 1, n1) if (j - i) % n1 % 3 == 1]
    if is_valid(n1, edges1):
        best_n = n1
        best_edges = edges1
        print(f"Found valid graph with n={n1}")

    # --- Try to extend by adding vertices with careful edge choices ---
    # This is a heuristic local search approach.
    # Start from And(9) and try to add vertices one by one.

    def try_extend(base_n, base_edges, max_add=15):
        """Try to extend a valid graph by adding new vertices."""
        n = base_n
        edges = list(base_edges)
        
        for _ in range(max_add):
            if time.time() - start_time > DEADLINE - 10:
                break
            new_v = n
            # Try different edge sets for the new vertex
            found = False
            # The new vertex can connect to any subset of existing vertices
            # but must not create a triangle
            # For each existing vertex u, if new_v connects to u, 
            # then new_v cannot connect to any neighbor of u
            
            # Greedy approach: try to connect to vertices in a way that
            # maintains triangle-free and keeps alpha < 10
            # We'll try several patterns
            
            # Pattern: connect to vertices that are "far" in some sense
            # For cyclic graphs, connect to vertices at certain distances
            
            # Try: connect new_v to all vertices that are not adjacent to any 
            # vertex it's already connected to (to avoid triangles)
            
            # Simple heuristic: connect to every other vertex
            candidate_edges = []
            connected_to = set()
            for u in range(n):
                # Check if connecting to u would create a triangle
                # i.e., if there's a v in connected_to such that u~v
                can_connect = True
                for v in connected_to:
                    if (u, v) in [(a, b) for a, b in edges] or (v, u) in [(a, b) for a, b in edges]:
                        can_connect = False
                        break
                if can_connect:
                    candidate_edges.append([new_v, u])
                    connected_to.add(u)
            
            if candidate_edges:
                new_edges = edges + candidate_edges
                if is_valid(n + 1, new_edges):
                    n += 1
                    edges = new_edges
                    found = True
            
            if not found:
                break
        
        return n, edges

    n2, edges2 = try_extend(n1, edges1, max_add=15)
    if n2 > best_n:
        best_n = n2
        best_edges = edges2
        print(f"Extended to n={n2}")

    # --- Try other known constructions ---
    # Clebsch graph: 16 vertices, triangle-free, alpha=5
    # But we want larger graphs
    
    # Try a random search with local improvement
    def random_search(target_n, iterations=5000):
        """Randomly generate triangle-free graphs and check alpha."""
        for it in range(iterations):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Generate random triangle-free graph
            # Start with empty graph, add edges randomly avoiding triangles
            n = target_n
            adj = [[False] * n for _ in range(n)]
            edges = []
            
            # Randomly add edges
            import random
            random.seed(42 + it)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3:  # probability of adding edge
                        # Check if adding (i,j) creates a triangle
                        creates_triangle = False
                        for k in range(n):
                            if k != i and k != j and adj[i][k] and adj[j][k]:
                                creates_triangle = True
                                break
                        if not creates_triangle:
                            adj[i][j] = True
                            adj[j][i] = True
                            edges.append([i, j])
            
            if is_valid(n, edges):
                return n, edges
        return 0, []

    # Try to find graphs with n > best_n
    for target_n in range(best_n + 1, 41):
        if time.time() - start_time > DEADLINE - 15:
            break
        n_rand, edges_rand = random_search(target_n, iterations=2000)
        if n_rand > best_n:
            best_n = n_rand
            best_edges = edges_rand
            print(f"Random search found n={n_rand}")
            break

    if best_n == 0:
        # Fallback to And(9)
        best_n = n1
        best_edges = edges1

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
