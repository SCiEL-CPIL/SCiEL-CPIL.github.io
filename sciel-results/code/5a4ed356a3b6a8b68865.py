"""Combined program for R(3,10) triangle-free graph construction."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 108  # safety margin before 120s limit

    best_n = 0
    best_edges = []

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_triangle_free(n, edges):
        adj = build_adj(n, edges)
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                common = adj[i] & adj[j]
                if common:
                    return False
        return True

    def alpha_size(n, adj, limit=10, deadline=None):
        """Return size of max independent set, capped: returns >=limit if found.
        Uses branch and bound; stops early if a set of size `limit` is found."""
        best = [0]
        # order vertices by degree descending for pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        # We'll work on a list-based recursion with bitsets for speed
        # Represent candidate set as a frozenset of vertices
        def branch(candidates, size):
            nonlocal best
            if deadline is not None and time.time() > deadline:
                return
            if size > best[0]:
                best[0] = size
                if best[0] >= limit:
                    return
            if not candidates:
                return
            # Upper bound
            if size + len(candidates) <= best[0]:
                return
            # Pick vertex with max degree within candidates
            cands = list(candidates)
            v = max(cands, key=lambda x: len(adj[x] & candidates))
            # Include v
            new_cands = candidates - {v} - adj[v]
            branch(new_cands, size + 1)
            if best[0] >= limit:
                return
            # Exclude v
            branch(candidates - {v}, size)

        branch(frozenset(range(n)), 0)
        return best[0]

    def is_valid(n, edges, deadline=None):
        if deadline is not None and time.time() > deadline:
            return False
        if not check_triangle_free(n, edges):
            return False
        adj = build_adj(n, edges)
        a = alpha_size(n, adj, limit=10, deadline=deadline)
        return a < 10

    def record(n, edges):
        nonlocal best_n, best_edges
        if n > best_n:
            best_n = n
            best_edges = list(edges)

    # --- Seed 1: Andrasfai And(9) on 26 vertices ---
    n1 = 26
    edges1 = [[i, j] for i in range(n1) for j in range(i + 1, n1) if (j - i) % n1 % 3 == 1]
    if is_valid(n1, edges1, DEADLINE - 5):
        record(n1, edges1)

    # --- Seed 2: Tutte-Coxeter graph on 30 vertices (3-regular, triangle-free, alpha=9) ---
    # The Tutte-Coxeter graph (Biggs-Smith) is the point-line incidence graph of GQ(2,2)?
    # Actually it's a 3-regular graph on 30 vertices. Let's use a known construction:
    # It can be generated as a circulant-like or via explicit adjacency.
    # Known adjacency for Tutte-Coxeter (30 vertices):
    # vertices 0..29, edges form a specific pattern.
    # One representation: it's the Levi graph of the generalized quadrangle of order (2,2)?
    # That has (2+1)(2*2+1)=15 points and 15 lines -> 30 vertices, 3-regular? No, that's (2+1)=3 regular.
    # Yes! GQ(2,2) has 15 points, 15 lines, each point on 3 lines, each line has 3 points.
    # So the incidence graph is 3-regular on 30 vertices, triangle-free (bipartite), and
    # its independence number is 15 (one part). That's too big!
    # Wait, alpha of a bipartite graph with parts of 15 is at least 15. So alpha>=15 >=10. Invalid.
    # So Tutte-Coxeter via GQ(2,2) is NOT valid. Let me reconsider.
    # Actually the Tutte-Coxeter graph is 3-regular on 30 vertices and IS triangle-free,
    # but its independence number is 9? Let me verify differently.
    # Hmm, a 3-regular graph on 30 vertices: by Turan, max independent set...
    # Actually, let me just use a known valid 30-vertex graph if I can construct one.
    # Let me skip this and try the circulant family for n up to 40.

    # --- Circulant family C_n(1 mod 3) for various n ---
    # Triangle-free when n not divisible by 3 (as analyzed).
    # For n=26, alpha=9 (valid). Let's test n=28, 29, 31, 32, 34, 35, 37, 38, 40.
    for n in [28, 29, 31, 32, 34, 35, 37, 38, 40]:
        if time.time() > DEADLINE - 15:
            break
        edges_c = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        if is_valid(n, edges_c, DEADLINE - 10):
            record(n, edges_c)
            print(f"Circulant valid at n={n}")

    # --- Local extension from best found so far ---
    def try_extend(base_n, base_edges, max_add=14):
        """Try to extend a valid graph by adding new vertices with smart edge choices."""
        n = base_n
        edges = list(base_edges)
        adj = build_adj(n, edges)

        for _ in range(max_add):
            if time.time() > DEADLINE - 10:
                break
            new_v = n
            # The new vertex can connect to any independent set in the current graph
            # (to avoid triangles, its neighbors must form an independent set).
            # To keep alpha < 10, we want the new vertex to be "useful":
            # connecting it to a large independent set reduces the alpha.
            # Strategy: find a maximal independent set S in current graph,
            # connect new_v to all of S. Then new_v's neighbors = S (independent),
            # so no triangle. The new alpha: any independent set in new graph
            # either avoids new_v (size <= old alpha) or includes new_v
            # (then can include no vertex of S, so size <= 1 + alpha(G - S)).
            # We need both < 10.

            # Find a large independent set in current graph
            S = greedy_independent_set(n, adj)
            if not S:
                break

            # Connect new_v to all vertices in S
            new_edges = edges + [[new_v, u] for u in S]
            if is_valid(n + 1, new_edges, DEADLINE - 8):
                n += 1
                edges = new_edges
                adj = build_adj(n, edges)
            else:
                # Try a smaller subset of S
                # Remove the vertex in S with highest degree (least useful)
                S_sorted = sorted(S, key=lambda x: -len(adj[x]))
                for remove_count in range(1, len(S)):
                    S_sub = [v for v in S if v not in S_sorted[:remove_count]]
                    if not S_sub:
                        break
                    new_edges2 = edges + [[new_v, u] for u in S_sub]
                    if is_valid(n + 1, new_edges2, DEADLINE - 8):
                        n += 1
                        edges = new_edges2
                        adj = build_adj(n, edges)
                        break
                else:
                    break

        return n, edges

    def greedy_independent_set(n, adj):
        """Greedy maximal independent set."""
        remaining = set(range(n))
        ind = set()
        while remaining:
            v = max(remaining, key=lambda x: -len(adj[x] & remaining))
            ind.add(v)
            remaining -= adj[v]
            remaining -= {v}
        return ind

    if best_n > 0:
        n_ext, edges_ext = try_extend(best_n, best_edges, max_add=14)
        record(n_ext, edges_ext)
        print(f"Extended to n={n_ext}")

    # --- Random search for larger n ---
    def random_search(target_n, iterations=3000):
        for it in range(iterations):
            if time.time() > DEADLINE - 10:
                break
            random.seed(12345 + it)
            adj = [set() for _ in range(target_n)]
            edges = []
            # Random triangle-free graph generation
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.25:
                        if not (adj[i] & adj[j]):
                            adj[i].add(j)
                            adj[j].add(i)
                            edges.append([i, j])
            # Quick alpha check
            a = alpha_size(target_n, adj, limit=10, deadline=DEADLINE - 8)
            if a < 10:
                return target_n, edges
        return 0, []

    for target_n in range(best_n + 1, 41):
        if time.time() > DEADLINE - 15:
            break
        n_r, edges_r = random_search(target_n, iterations=2000)
        if n_r > best_n:
            record(n_r, edges_r)
            print(f"Random found n={n_r}")
            break

    if best_n == 0:
        best_n = n1
        best_edges = edges1

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
