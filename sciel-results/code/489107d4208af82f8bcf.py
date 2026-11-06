"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_independent_set_10(n, adj_mask):
    """Return True iff there exists an independent set of size 10.
    Uses bitmask branch-and-bound with greedy coloring upper bound on complement."""
    all_mask = (1 << n) - 1

    # Greedy coloring of complement (i.e., independent-set bound on original)
    # We want to upper-bound the size of a maximum independent set.
    # Equivalent to: color the complement graph greedily; chromatic number >= alpha.
    # Actually for independent set in G, we need coloring of G (not complement).
    # Wait: alpha(G) = chi(complement(G)). So to bound alpha(G), we color complement(G).
    # But complement coloring is expensive. Instead, use a simple bound:
    # Greedy coloring of G gives upper bound on omega(G), not alpha(G).
    # For alpha, use: repeatedly pick a vertex, remove it and its non-neighbors? No.
    # Simple bound: alpha <= n. Better: use a greedy packing.
    # Let's use a simple recursive bound with memoization-free approach.

    best_found = 0  # we only need to know if >= 10

    def dfs(candidates, depth):
        nonlocal best_found
        if depth >= 10:
            return True
        if not candidates:
            return False
        # Upper bound: depth + popcount(candidates)
        if depth + _popcount(candidates) < 10:
            return False

        # Pick vertex with minimum degree in induced subgraph (for independent set)
        # Actually, for finding large independent set, pick vertex with few neighbors
        # in candidates to maximize remaining candidates after including it.
        # But for pruning, pick vertex that maximizes the bound.
        # Use: pick vertex with maximum degree in complement within candidates
        # (i.e., minimum non-neighborhood), so excluding it removes few from candidates.
        # Simpler: just pick first vertex or use a heuristic.

        # Heuristic: pick vertex v in candidates that minimizes |N(v) ∩ candidates|
        # This way, including v leaves many candidates, but excluding v also leaves many.
        # Actually for independent set search, we want to branch on a vertex where
        # including it prunes a lot (high degree) OR where the bound is tight.
        # Let's pick the vertex with the highest degree in the induced subgraph on candidates.
        # High degree means including it removes many candidates (good pruning for include branch).

        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            v = _ctz(c)
            c &= c - 1
            # degree of v in induced subgraph on candidates
            deg = _popcount(adj_mask[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = v

        v = best_v
        v_bit = 1 << v

        # Branch 1: include v
        # New candidates: those not adjacent to v
        new_cand = candidates & ~adj_mask[v] & ~v_bit
        if dfs(new_cand, depth + 1):
            return True

        # Branch 2: exclude v
        new_cand = candidates & ~v_bit
        if dfs(new_cand, depth):
            return True

        return False

    return dfs(all_mask, 0)


def _popcount(x):
    return bin(x).count('1')


def _ctz(x):
    """Count trailing zeros (index of lowest set bit)."""
    if x == 0:
        return -1
    return (x & -x).bit_length() - 1


def _is_triangle_free(n, adj_mask):
    """Check if graph is triangle-free using adjacency bitmasks."""
    for u in range(n):
        au = adj_mask[u]
        # For each pair of neighbors of u, check if they are adjacent
        # This would be O(n^3) worst case. Better: for each edge (u,v), check N(u) & N(v)
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_mask[u] >> v) & 1:
                # Check if any common neighbor
                if adj_mask[u] & adj_mask[v]:
                    return False
    return True


def _validate(n, edges):
    """Check both conditions."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    if not _is_triangle_free(n, adj_mask):
        return False
    if _has_independent_set_10(n, adj_mask):
        return False
    return True


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Seed: Andrásfai graph on 26 vertices
    # And(9): blow-up of C9 with parts of size 3, edges between consecutive parts
    # Actually, let's use a known triangle-free graph with small independence number.
    # The Andrásfai graph And(k) has n=k^2, is triangle-free, and alpha = k^2 - k + 1? No.
    # And(5) has n=25, alpha=... Let me just use a random search.

    best_n = 1
    best_edges = []

    # Try known small constructions first
    # Andrásfai graph: partition into k groups of size k, connect group i to group (i+1) mod k
    # This is triangle-free. Independence number?
    # For k=5: n=25, groups of 5. Edges between consecutive groups.
    # Independent set: can pick at most one from each "consecutive" structure...
    # Actually alpha(And(k)) = k^2 - k + 1? For k=5, that's 21. Too large.
    # We need alpha < 10.

    # Let's try random triangle-free graphs with local search.

    random.seed(42)

    for target_n in range(40, 25, -1):
        if time.time() > deadline - 5:
            break

        found = False
        max_attempts = 200
        attempts = 0

        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 5:
                break

            n = target_n
            adj_mask = [0] * n
            edges = []

            # Generate random triangle-free graph
            # Randomly order potential edges and add with probability p
            # Check triangle-freeness on the fly

            # Use a random permutation of all edges
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)

            p = 0.2 + 0.3 * random.random()  # density between 0.2 and 0.5

            for u, v in all_edges:
                if random.random() < p:
                    # Check if adding (u,v) creates a triangle
                    if adj_mask[u] & adj_mask[v]:
                        continue
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.append([u, v])

            # Check independent set
            if not _has_independent_set_10(n, adj_mask):
                best_n = n
                best_edges = edges
                found = True
                break

        if found:
            break

    # If nothing found above best_n, return seed
    if best_n < 26:
        # Fallback: simple graph on 26 vertices
        n = 26
        edges = []
        adj_mask = [0] * n
        # Use a random triangle-free graph as fallback
        random.seed(123)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if random.random() < 0.3:
                if not (adj_mask[u] & adj_mask[v]):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.append([u, v])
        # Verify
        if _is_triangle_free(n, adj_mask) and not _has_independent_set_10(n, adj_mask):
            best_n = n
            best_edges = edges
        else:
            # Absolute fallback: single vertex
            best_n = 1
            best_edges = []

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
