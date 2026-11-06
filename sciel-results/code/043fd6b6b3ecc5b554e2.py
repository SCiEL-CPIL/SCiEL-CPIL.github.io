"""Stronger algorithm: iterative randomized construction of C4-free graphs with high min-degree."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_n = 39
    best_edges = []

    def is_c4_free(adj_set, n):
        """Check if graph with adjacency sets is C4-free.
        A C4 exists iff two vertices have 2+ common neighbors."""
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj_set[u] & adj_set[v]) >= 2:
                    return False
        return True

    def min_degree(adj_set, n):
        return min(len(adj_set[u]) for u in range(n))

    def to_edges(adj_set, n):
        edges = []
        for u in range(n):
            for v in adj_set[u]:
                if v > u:
                    edges.append([u, v])
        return edges

    def try_random_construction(n, seed, max_iter=2000):
        """Try to build a C4-free graph on n vertices with min-degree >= n-39."""
        rng = random.Random(seed)
        target_min_deg = n - 39
        if target_min_deg < 0:
            target_min_deg = 0

        # Strategy: start with a random graph and iteratively fix violations
        # We want min-degree >= target_min_deg and no C4.

        # Approach: random greedy - add edges that don't create C4, trying to boost low-degree vertices
        for attempt in range(50):
            # Start with a random initial graph
            adj = [set() for _ in range(n)]

            # Try to build a graph where each vertex has degree close to n-1-target_min_deg complement
            # We want complement to have max degree <= 38 (no K1,39)
            # Equivalently, original graph min-degree >= n-39

            # Random initialization: for each pair, include edge with some probability
            # To get min-degree near n-39, we want to exclude about 38 edges per vertex on average
            # So include edge with prob ~ (n-1-38)/(n-1) for large n

            # Better approach: start dense, then remove edges to eliminate C4s while maintaining min-degree
            # Or start sparse and add edges

            # Let's try: start with complete graph, remove edges to break C4s
            # This is equivalent to: find a C4-free graph that is as dense as possible
            # with the constraint min-degree >= n-39

            # Alternative: random greedy edge addition
            # For each vertex, we need degree >= target_min_deg
            # We add edges randomly, checking C4 constraint

            edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
            rng.shuffle(edges_to_try)

            adj = [set() for _ in range(n)]
            edge_count = 0

            for (u, v) in edges_to_try:
                if time.time() > deadline:
                    return None, None
                # Check if adding edge (u,v) creates a C4
                # A C4 would be formed if there exist x,y such that
                # u-x, x-v are edges (path u-x-v) AND u-y, y-v are edges (path u-y-v)
                # with x != y, and x,y not equal to u,v
                # More precisely: adding (u,v) creates C4 iff |N(u) ∩ N(v)| >= 2
                # because any two common neighbors x,y give u-x-v-y-u cycle
                common = adj[u] & adj[v]
                if len(common) < 2:
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_count += 1

            md = min_degree(adj, n)
            if md >= target_min_deg:
                return n, to_edges(adj, n)

        return None, None

    # Try increasing n from high to low
    # Start from n=47 down to n=40
    # For each n, try many random seeds
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        target = n - 39
        found = False
        for seed in range(10000):
            if time.time() > deadline:
                break
            res_n, res_edges = try_random_construction(n, seed)
            if res_n is not None:
                # Verify
                adj = [set() for _ in range(n)]
                for e in res_edges:
                    adj[e[0]].add(e[1])
                    adj[e[1]].add(e[0])
                if is_c4_free(adj, n) and min_degree(adj, n) >= target:
                    if n > best_n:
                        best_n = n
                        best_edges = res_edges
                    found = True
                    break  # Found a valid graph for this n, move to next (higher) n

    # If we found n=47 or 46, great. Otherwise try more aggressively for lower n.
    # Also try a different strategy: start from a known structure.

    # Alternative strategy: use a random regular-like graph
    # For each n, try to construct via different methods

    if best_n <= 39:
        # Try a more systematic approach: for each n, use simulated annealing-like
        # or try structured constructions

        for n in range(47, 39, -1):
            if time.time() > deadline:
                break
            target = n - 39
            # Try multiple random restarts with different edge orderings
            for seed in range(50000):
                if time.time() > deadline:
                    break
                rng = random.Random(seed)
                adj = [set() for _ in range(n)]

                # Random edge ordering, greedy add if no C4
                edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
                rng.shuffle(edges_to_try)

                for (u, v) in edges_to_try:
                    common = adj[u] & adj[v]
                    if len(common) < 2:
                        adj[u].add(v)
                        adj[v].add(u)

                md = min_degree(adj, n)
                if md >= target:
                    if n > best_n:
                        best_n = n
                        best_edges = to_edges(adj, n)
                    break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
