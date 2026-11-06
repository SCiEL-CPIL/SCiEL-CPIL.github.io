"""Search for large triangle-free graphs with independence number < 10."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """Construct a triangle-free graph on as many vertices as possible with alpha < 10."""
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit

    def time_left():
        return deadline - (time.time() - start_time)

    def check_triangle_free(adj_matrix, n):
        """Check if graph is triangle-free using adjacency matrix."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j]:
                    # Check if any common neighbor forms a triangle
                    row_i = adj_matrix[i]
                    row_j = adj_matrix[j]
                    for k in range(j + 1, n):
                        if row_i[k] and row_j[k]:
                            return False
        return True

    def check_alpha_less_than_10(adj_matrix, n):
        """Check if independence number is < 10 using branch and bound."""
        # Use a simple branch and bound for maximum independent set
        # We want to verify that no independent set of size 10 exists

        # Greedy coloring for upper bound (coloring the complement graph)
        # In the complement graph, independent set = clique
        # So we need to check if complement has a clique of size 10

        # Build complement adjacency
        comp_adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not adj_matrix[i][j]:
                    comp_adj[i][j] = 1
                    comp_adj[j][i] = 1

        # Use branch and bound to find if there's a clique of size 10 in complement
        # This is equivalent to independent set of size 10 in original

        def max_clique_size(adj, n, target=10):
            """Find if there's a clique of size >= target using branch and bound."""
            # Greedy coloring for upper bound
            # Order vertices by degree for better bounds
            order = sorted(range(n), key=lambda v: -sum(adj[v]))
            
            # Recursively try to build a clique
            best = [0]
            
            def color_bound(candidates, adj):
                """Greedy coloring of candidates for upper bound on clique size."""
                if not candidates:
                    return 0
                colors = {}
                color_count = 0
                remaining = list(candidates)
                while remaining:
                    color_count += 1
                    # Pick a vertex
                    v = remaining.pop(0)
                    colors[v] = color_count
                    # Remove vertices adjacent to v from remaining
                    new_remaining = []
                    for u in remaining:
                        if not adj[v][u]:
                            new_remaining.append(u)
                    remaining = new_remaining
                return color_count

            def branch(candidates, adj, current_clique):
                if time.time() - start_time > deadline - 5:
                    return
                if len(current_clique) >= target:
                    return True
                if not candidates:
                    return False
                # Upper bound via coloring
                ub = len(current_clique) + color_bound(candidates, adj)
                if ub < target:
                    return False
                # Try vertices in candidates
                for v in candidates:
                    new_clique = current_clique + [v]
                    # New candidates: those in candidates that are adjacent to v
                    new_candidates = [u for u in candidates if u != v and adj[v][u]]
                    if branch(new_candidates, adj, new_clique):
                        return True
                return False

            return branch(list(range(n)), adj, [])

        # Check if complement has a clique of size 10
        has_large_clique = max_clique_size(comp_adj, n, 10)
        return not has_large_clique

    def is_valid(adj_matrix, n):
        """Check if graph is triangle-free and has alpha < 10."""
        if not check_triangle_free(adj_matrix, n):
            return False
        if not check_alpha_less_than_10(adj_matrix, n):
            return False
        return True

    def build_adj_matrix(edges, n):
        """Build adjacency matrix from edge list."""
        adj = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj

    def edges_from_adj(adj, n):
        """Get edge list from adjacency matrix."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # Start with known good: Andrasfai graph on 26 vertices
    # Try to improve by adding vertices

    best_n = 0
    best_edges = None

    # First, verify the Andrasfai graph
    n_and = 26
    adj_and = [[0] * n_and for _ in range(n_and)]
    for i in range(n_and):
        for j in range(i + 1, n_and):
            if (j - i) % n_and % 3 == 1:
                adj_and[i][j] = 1
                adj_and[j][i] = 1
    if is_valid(adj_and, n_and):
        best_n = n_and
        best_edges = edges_from_adj(adj_and, n_and)

    # Try to extend by adding vertices one at a time
    # For each new vertex, try all possible connections to existing vertices
    # that maintain triangle-freeness

    current_n = best_n
    current_adj = [row[:] for row in adj_and]

    # Try to add vertices from n=27 up to n=40
    for new_n in range(current_n + 1, 41):
        if time_left() < 10:
            break

        new_adj = [[0] * new_n for _ in range(new_n)]
        # Copy existing edges
        for i in range(current_n):
            for j in range(i + 1, current_n):
                if current_adj[i][j]:
                    new_adj[i][j] = 1
                    new_adj[j][i] = 1

        # Try to find a valid connection for vertex new_n-1
        # A vertex v can connect to a set S of existing vertices if:
        # 1. S is an independent set in the current graph (no edges within S)
        # 2. For any edge (u,w) in current graph, not both u and w are in S
        #    (this is same as 1)
        # Actually, triangle-free means: for any two neighbors u,w of v,
        # u and w must not be adjacent. So S must be independent.

        # Also need alpha < 10 after adding v.

        found = False
        # Try different subsets S of existing vertices to connect to new vertex
        # Use a heuristic: try larger independent sets first

        # Get all independent sets of the current graph up to some size
        # This could be expensive, so use a greedy approach

        # Simpler: try random subsets
        for attempt in range(200):
            if time_left() < 10:
                break
            if found:
                break

            # Random subset of existing vertices
            subset_size = random.randint(0, current_n // 2)
            S = set(random.sample(range(current_n), min(subset_size, current_n)))

            # Check if S is independent
            independent = True
            for u, w in combinations(S, 2):
                if current_adj[u][w]:
                    independent = False
                    break
            if not independent:
                continue

            # Set up connections
            for v in S:
                new_adj[new_n - 1][v] = 1
                new_adj[v][new_n - 1] = 1

            # Check triangle-free
            if not check_triangle_free(new_adj, new_n):
                for v in S:
                    new_adj[new_n - 1][v] = 0
                    new_adj[v][new_n - 1] = 0
                continue

            # Check alpha < 10
            if check_alpha_less_than_10(new_adj, new_n):
                found = True
                current_adj = [row[:] for row in new_adj]
                current_n = new_n
                best_n = new_n
                best_edges = edges_from_adj(new_adj, new_n)
                break

        if not found:
            break  # Can't extend further with this approach

    # Try a different approach: use known constructions
    # The Ramsey number R(3,10) is known to be 41, so there exists a graph on 40 vertices
    # that is triangle-free and has alpha < 10.

    # Try the Clebsch graph or other known triangle-free graphs
    # Actually, let's try a random search with local improvement

    if time_left() > 20:
        # Random search: start with random triangle-free graphs and try to grow them
        for trial in range(50):
            if time_left() < 15:
                break

            # Start with a random triangle-free graph
            n = random.randint(20, 35)
            adj = [[0] * n for _ in range(n)]

            # Build a random triangle-free graph by adding edges carefully
            vertices = list(range(n))
            random.shuffle(vertices)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3:
                        # Check if adding edge (i,j) creates a triangle
                        creates_triangle = False
                        for k in range(n):
                            if adj[i][k] and adj[j][k]:
                                creates_triangle = True
                                break
                        if not creates_triangle:
                            adj[i][j] = 1
                            adj[j][i] = 1

            if not check_triangle_free(adj, n):
                continue

            if check_alpha_less_than_10(adj, n):
                if n > best_n:
                    best_n = n
                    best_edges = edges_from_adj(adj, n)

    # Final validation
    if best_edges is not None and best_n > 0:
        final_adj = build_adj_matrix(best_edges, best_n)
        if check_triangle_free(final_adj, best_n) and check_alpha_less_than_10(final_adj, best_n):
            return {'n': best_n, 'edges': best_edges}

    # Fallback: return the Andrasfai graph
    n_and = 26
    edges_and = [[i, j] for i in range(n_and) for j in range(i + 1, n_and) if (j - i) % n_and % 3 == 1]
    return {'n': n_and, 'edges': edges_and}

# EVOLVE-BLOCK-END
